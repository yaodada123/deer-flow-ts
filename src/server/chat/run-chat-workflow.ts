import { randomUUID } from "node:crypto";

import { Annotation, END, START, StateGraph, type LangGraphRunnableConfig } from "@langchain/langgraph";

import type { ChatRequest, SkillId } from "../schemas.js";
import type { LlmConfig } from "../llm/env-models.js";
import { loadLlmConfig } from "../llm/env-models.js";
import { OpenAICompatibleClient, type OpenAIChatMessage } from "../llm/openai-compatible.js";
import { formatSkillPromptBlock } from "../skills/prompt.js";
import { getSkills } from "../skills/registry.js";
import { selectSkills } from "../skills/selector.js";
import { academicResultsToToolResult, academicSearch, formatAcademicResults } from "../tools/academic-search.js";
import { retrieveResources } from "../tools/resources.js";
import { webSearch } from "../tools/web-search.js";
import {
  buildFallbackPlan,
  buildPlannerEditPrompt,
  buildPlannerPrompt,
  buildReporterPrompt,
  ensureTopicPlanFields,
  safeParsePlan,
} from "../workflow.js";
import type { ThreadStore } from "../runtime/thread-store.js";
import type { WorkflowState } from "../runtime/types.js";
import type { TraceRecorder } from "../trace/recorder.js";
import type { TraceStatus } from "../trace/types.js";

type ChatEvent = {
  type: string;
  data: Record<string, unknown>;
};

type ToolCall = {
  type: "tool_call";
  id: string;
  name: string;
  args: Record<string, unknown>;
};

type ToolCallChunk = {
  type: "tool_call_chunk";
  index: number;
  id: string;
  name: string;
  args: string;
};

function newMessageId(): string {
  return `run-${randomUUID()}`;
}

function newToolCallId(): string {
  return `call_${randomUUID().replace(/-/g, "")}`;
}

function stripThinkTags(text: string): string {
  return text.replaceAll("<think>", "").replaceAll("</think>", "");
}

function pickLlmConfig(params: { enableDeepThinking: boolean }): LlmConfig | null {
  if (params.enableDeepThinking) {
    return loadLlmConfig("reasoning") ?? loadLlmConfig("basic");
  }
  return loadLlmConfig("basic") ?? loadLlmConfig("reasoning");
}

function normalizeUserContent(request: ChatRequest): string {
  const lastUser = [...request.messages].reverse().find((m) => m.role === "user") ?? request.messages.at(-1);
  if (!lastUser) return "";
  if (typeof lastUser.content === "string") return lastUser.content;
  return lastUser.content.map((c) => c.text ?? "").join("");
}

function normalizeChatMessageContent(content: ChatRequest["messages"][number]["content"]): string {
  if (typeof content === "string") return content;
  return content.map((c) => c.text ?? "").join("");
}

function normalizeChatMessages(request: ChatRequest): OpenAIChatMessage[] {
  return request.messages
    .filter((m) => m.role === "user" || m.role === "assistant")
    .map((m) => ({
      role: m.role as "user" | "assistant",
      content: normalizeChatMessageContent(m.content),
    }))
    .filter((m) => m.content.trim().length > 0);
}

function plainChatFallback(locale: string): string {
  if (isChineseLocale(locale)) {
    return "我可以进行普通对话，但当前未配置模型。请配置 BASIC_MODEL__MODEL 和 BASIC_MODEL__API_KEY 后再试。";
  }
  return "I can chat normally, but no model is configured. Please set BASIC_MODEL__MODEL and BASIC_MODEL__API_KEY, then try again.";
}

function plainChatSystemPrompt(locale: string): string {
  return [
    "You are ScholarFlow in normal chat mode.",
    "Answer the user's message conversationally and helpfully.",
    "Do not create a research plan, do not run academic research workflow, and do not promise evidence retrieval or a full research report.",
    "If the user asks for a research report, literature review, source-grounded investigation, or evidence retrieval, explain that they can enable Research Mode to run the full workflow, while still providing a concise direct answer if possible.",
    `Respond in the user's language. Locale: ${locale}.`,
  ].join(" ");
}

function resolveActiveSkills(params: {
  request: ChatRequest;
  query: string;
  existing: WorkflowState | undefined;
  isFeedback: boolean;
}): { activeSkills: SkillId[]; reason: string } {
  const { request, query, existing, isFeedback } = params;
  if (!request.enable_skills) return { activeSkills: [], reason: "skills disabled" };
  if (request.selected_skills.length) return { activeSkills: request.selected_skills, reason: "manual selection" };
  if (isFeedback && existing) {
    return {
      activeSkills: existing.activeSkills,
      reason: existing.skillSelectionReason ?? "preserved from pending plan",
    };
  }
  const selected = selectSkills({ query, resources: request.resources });
  return {
    activeSkills: selected.activeSkills.map((skill) => skill.id),
    reason: selected.reason,
  };
}

function hasSkill(state: Pick<WorkflowState, "activeSkills">, skillId: SkillId): boolean {
  return state.activeSkills.includes(skillId);
}

function academicSearchLimit(state: Pick<WorkflowState, "activeSkills" | "maxSearchResults">): number {
  if (hasSkill(state, "systematic-literature-review")) {
    return Math.min(20, Math.max(10, state.maxSearchResults));
  }
  return state.maxSearchResults;
}

type CoordinatorDecision =
  | { action: "direct_response"; message: string }
  | { action: "handoff_to_planner" };

function isChineseLocale(locale: string): boolean {
  return locale.toLowerCase().startsWith("zh");
}

function isLikelyChitChat(userText: string): boolean {
  const trimmed = userText.trim();
  if (!trimmed) return true;

  const lower = trimmed.toLowerCase();

  if (/^(hi|hello|hey|good\s+(morning|afternoon|evening))[^a-z0-9]*$/i.test(trimmed)) return true;
  if (/^(你好|您好|嗨|哈喽|早上好|上午好|中午好|下午好|晚上好|在吗)[！!。,.?？]*$/.test(trimmed)) return true;

  if (lower.length <= 50) {
    if (/(what\s+can\s+you\s+do|who\s+are\s+you|your\s+name)/.test(lower)) return true;
    if (/(你是谁|你叫什么|你能做什么|你会什么|你是什么)/.test(trimmed)) return true;
  }

  return false;
}

function defaultChitChatReply(params: { locale: string; userText: string }): string {
  if (isChineseLocale(params.locale)) {
    return "你好！我是 ScholarFlow，一个学术研究助手。你可以给我一个研究问题，或上传论文/笔记让我帮你规划研究、整理证据并生成报告。";
  }
  return "Hi! I'm ScholarFlow, an academic research assistant. Share a research question or upload papers/notes, and I can help plan the research, retrieve evidence, and draft a structured report.";
}

function tryParseJsonObject(text: string): Record<string, unknown> | null {
  try {
    const parsed = JSON.parse(text) as unknown;
    if (!parsed || typeof parsed !== "object" || Array.isArray(parsed)) return null;
    return parsed as Record<string, unknown>;
  } catch {
    return null;
  }
}

function extractFirstJsonObject(text: string): string | null {
  const start = text.indexOf("{");
  if (start < 0) return null;
  const end = text.lastIndexOf("}");
  if (end <= start) return null;
  return text.slice(start, end + 1);
}

function parseCoordinatorDecision(text: string): CoordinatorDecision | null {
  const trimmed = text.trim();
  const direct = tryParseJsonObject(trimmed) ?? tryParseJsonObject(extractFirstJsonObject(trimmed) ?? "");
  if (!direct) return null;

  const action = direct.action;
  if (action === "handoff_to_planner") return { action: "handoff_to_planner" };
  if (action !== "direct_response") return null;

  const message = direct.message;
  if (typeof message !== "string" || !message.trim()) return null;

  return { action: "direct_response", message };
}

function extractObservationSources(observations: string[]): Array<{ title: string; uri: string }> {
  const sources: Array<{ title: string; uri: string }> = [];
  const seen = new Set<string>();
  const sourcePattern = /^- \[\d+\] (.+?) \((https?:\/\/[^)]+)\)/gm;

  for (const observation of observations) {
    for (const match of observation.matchAll(sourcePattern)) {
      const title = match[1]?.trim();
      const uri = match[2]?.trim();
      if (!title || !uri || seen.has(uri)) continue;
      seen.add(uri);
      sources.push({ title, uri });
    }
  }

  return sources;
}

async function decideCoordinatorAction(params: {
  llm: OpenAICompatibleClient | null;
  locale: string;
  userText: string;
  signal?: AbortSignal;
}): Promise<CoordinatorDecision> {
  if (!params.llm) {
    if (isLikelyChitChat(params.userText)) {
      return {
        action: "direct_response",
        message: defaultChitChatReply({ locale: params.locale, userText: params.userText }),
      };
    }
    return { action: "handoff_to_planner" };
  }

  const system =
    "You are ScholarFlow Coordinator, an academic research assistant. Decide whether to respond directly or hand off to the planner. " +
    "You MUST respond directly for greetings, small talk, identity/capability questions (e.g., 'who are you', 'what can you do'). " +
    "You MUST hand off for academic research, literature review, factual analysis, source-grounded writing, or information requests. " +
    'Output ONLY valid JSON (no markdown). Schema: {"action":"direct_response"|"handoff_to_planner","message"?:string}. ' +
    "When action=direct_response, message is required and must be in the user's language.";

  const user = `Locale: ${params.locale}\nUser message: ${params.userText}`;

  let out = "";
  try {
    for await (const delta of params.llm.streamChatCompletions({
      messages: [
        { role: "system", content: system },
        { role: "user", content: user },
      ],
      ...(params.signal ? { signal: params.signal } : {}),
    })) {
      if (delta.content) out += delta.content;
      if (delta.finishReason) break;
    }
  } catch {
    return { action: "handoff_to_planner" };
  }

  return parseCoordinatorDecision(out) ?? { action: "handoff_to_planner" };
}

function makeBaseState(params: {
  request: ChatRequest;
  threadId: string;
  query: string;
  activeSkills: SkillId[];
  skillSelectionReason: string;
}): WorkflowState {
  const { request, threadId, query, activeSkills, skillSelectionReason } = params;
  return {
    threadId,
    locale: request.locale,
    researchTopic: query,
    messages: [{ role: "user", content: query }],
    resources: request.resources,
    observations: [],
    planIterations: 0,
    currentPlan: null,
    backgroundInvestigationResults: null,
    enableBackgroundInvestigation: request.enable_background_investigation,
    enableWebSearch: request.enable_web_search,
    maxPlanIterations: request.max_plan_iterations,
    maxStepNum: request.max_step_num,
    maxSearchResults: request.max_search_results,
    autoAcceptedPlan: request.auto_accepted_plan,
    reportStyle: request.report_style,
    activeSkills,
    skillSelectionReason,
  };
}

function contentPreview(content: unknown): Record<string, unknown> {
  if (typeof content !== "string") return {};
  return {
    content_length: content.length,
    content_preview: content.slice(0, 300),
  };
}

function summarizeToolResult(content: unknown): Record<string, unknown> {
  if (typeof content !== "string") return {};
  let count: number | undefined;
  try {
    const parsed = JSON.parse(content) as unknown;
    if (Array.isArray(parsed)) count = parsed.length;
  } catch {
    // Ignore non-JSON tool output.
  }
  return {
    content_length: content.length,
    content_preview: content.slice(0, 300),
    ...(count != null ? { item_count: count } : {}),
  };
}

function summarizeNodeOutput(output: unknown): Record<string, unknown> {
  if (!output || typeof output !== "object") return {};
  const obj = output as Record<string, unknown>;
  return {
    keys: Object.keys(obj),
    ...(Array.isArray(obj.observations) ? { observations: obj.observations.length } : {}),
    ...(obj.done ? { done: obj.done } : {}),
    ...(typeof obj.plannerShouldInterrupt === "boolean" ? { plannerShouldInterrupt: obj.plannerShouldInterrupt } : {}),
  };
}

async function tracedNode<TResult>(
  trace: TraceRecorder | undefined,
  params: { spanId: string; name: string; agent: string; input?: unknown; setCurrentSpan: (spanId: string | undefined) => void; currentSpan: string | undefined },
  fn: () => Promise<TResult>,
): Promise<TResult> {
  trace?.spanStarted({ spanId: params.spanId, name: params.name, agent: params.agent, input: params.input });
  const previousSpan = params.currentSpan;
  params.setCurrentSpan(params.spanId);
  try {
    const result = await fn();
    trace?.spanEnded({ spanId: params.spanId, name: params.name, agent: params.agent, status: "ok", output: summarizeNodeOutput(result) });
    return result;
  } catch (e) {
    trace?.spanEnded({ spanId: params.spanId, name: params.name, agent: params.agent, status: "error", error: e });
    throw e;
  } finally {
    params.setCurrentSpan(previousSpan);
  }
}

async function* runPlainChatResponse(params: {
  request: ChatRequest;
  llm: OpenAICompatibleClient | null;
  incomingText: string;
  signal?: AbortSignal;
  trace?: TraceRecorder;
}): AsyncIterable<ChatEvent> {
  const { request, llm, incomingText, signal, trace } = params;
  const spanId = `span_plain_chat_${randomUUID()}`;
  const id = newMessageId();

  const emit = (data: Record<string, unknown>): ChatEvent => {
    trace?.message({
      spanId,
      agent: "coordinator",
      metadata: {
        id: data.id,
        role: data.role,
        finish_reason: data.finish_reason,
        ...contentPreview(data.content),
      },
    });
    return { type: "message_chunk", data };
  };

  trace?.spanStarted({
    spanId,
    name: "chat.plain_response",
    agent: "coordinator",
    input: { resources: request.resources.length },
  });

  try {
    if (!llm) {
      yield emit({
        thread_id: request.thread_id,
        id,
        agent: "coordinator",
        role: "assistant",
        content: plainChatFallback(request.locale),
      });
      yield emit({
        thread_id: request.thread_id,
        id,
        agent: "coordinator",
        role: "assistant",
        finish_reason: "stop",
      });
      trace?.spanEnded({ spanId, name: "chat.plain_response", agent: "coordinator", status: "ok" });
      return;
    }

    const chatMessages = normalizeChatMessages(request);
    if (chatMessages.length === 0 && incomingText.trim()) {
      chatMessages.push({ role: "user", content: incomingText });
    }

    const resourceNotice: OpenAIChatMessage[] = request.resources.length
      ? [
          {
            role: "system",
            content:
              "The user attached resources, but normal chat mode does not retrieve or inspect resource contents. If the request depends on those materials, ask the user to enable Research Mode.",
          },
        ]
      : [];

    for await (const delta of llm.streamChatCompletions({
      messages: [
        { role: "system", content: plainChatSystemPrompt(request.locale) },
        ...resourceNotice,
        ...chatMessages,
      ],
      ...(signal ? { signal } : {}),
    })) {
      if (delta.content) {
        const cleaned = stripThinkTags(delta.content);
        if (cleaned) {
          yield emit({
            thread_id: request.thread_id,
            id,
            agent: "coordinator",
            role: "assistant",
            content: cleaned,
          });
        }
      }
      if (delta.finishReason) break;
    }

    yield emit({
      thread_id: request.thread_id,
      id,
      agent: "coordinator",
      role: "assistant",
      finish_reason: "stop",
    });
    trace?.spanEnded({ spanId, name: "chat.plain_response", agent: "coordinator", status: "ok" });
  } catch (e) {
    trace?.spanEnded({ spanId, name: "chat.plain_response", agent: "coordinator", status: "error", error: e });
    throw e;
  }
}

function upsertThreadState(params: {
  store: ThreadStore;
  request: ChatRequest;
  query: string;
  isFeedback: boolean;
  activeSkills: SkillId[];
  skillSelectionReason: string;
}): WorkflowState {
  const threadId = params.request.thread_id;
  const existing = params.store.get(threadId);
  if (!existing) {
    const created = makeBaseState({
      request: params.request,
      threadId,
      query: params.query,
      activeSkills: params.activeSkills,
      skillSelectionReason: params.skillSelectionReason,
    });
    params.store.set(created);
    return created;
  }

  const merged: WorkflowState = {
    ...existing,
    locale: params.request.locale,
    enableBackgroundInvestigation: params.request.enable_background_investigation,
    enableWebSearch: params.request.enable_web_search,
    maxPlanIterations: params.request.max_plan_iterations,
    maxStepNum: params.request.max_step_num,
    maxSearchResults: params.request.max_search_results,
    autoAcceptedPlan: params.request.auto_accepted_plan,
    reportStyle: params.request.report_style,
    activeSkills: params.activeSkills,
    skillSelectionReason: params.skillSelectionReason,
    resources: params.request.resources.length ? params.request.resources : existing.resources,
    ...(params.isFeedback
      ? {}
      : {
          researchTopic: params.query,
          messages: [...existing.messages, { role: "user", content: params.query }],
          observations: [],
          planIterations: 0,
          currentPlan: null,
          backgroundInvestigationResults: null,
        }),
  };
  params.store.set(merged);
  return merged;
}

type PlanningGraphState = WorkflowState & {
  incomingText: string;
  interruptFeedback: ChatRequest["interrupt_feedback"];
  isFeedback: boolean;
  coordinatorAction: "direct_response" | "handoff_to_planner" | null;
  plannerShouldInterrupt: boolean;
  done: "none" | "direct_response" | "interrupt_ready";
};

export async function* runChatWorkflow(params: {
  request: ChatRequest;
  store: ThreadStore;
  signal?: AbortSignal;
  trace?: TraceRecorder;
}): AsyncIterable<ChatEvent> {
  const { request, store, signal, trace } = params;
  const incomingText = normalizeUserContent(request);
  let currentSpanId: string | undefined;
  let runStatus: TraceStatus = "ok";
  let runEndReason = "completed";

  const isFeedback = request.interrupt_feedback === "accepted" || request.interrupt_feedback === "edit_plan";
  const existing = store.get(request.thread_id);
  const query = isFeedback ? (existing?.researchTopic ?? incomingText) : incomingText;
  const skillSelection = resolveActiveSkills({ request, query, existing, isFeedback });

  trace?.runStarted({
    query: incomingText,
    resources: request.resources.length,
    enable_web_search: request.enable_web_search,
    enable_background_investigation: request.enable_background_investigation,
    interrupt_feedback: request.interrupt_feedback,
    workflow_mode: request.workflow_mode,
    active_skills: skillSelection.activeSkills,
    skill_selection_reason: skillSelection.reason,
  });

  let state = upsertThreadState({
    store,
    request,
    query,
    isFeedback,
    activeSkills: skillSelection.activeSkills,
    skillSelectionReason: skillSelection.reason,
  });

  const llmCfg = pickLlmConfig({ enableDeepThinking: request.enable_deep_thinking });
  const llm = llmCfg ? new OpenAICompatibleClient(llmCfg) : null;

  try {
  if (request.workflow_mode === "chat" && !isFeedback) {
    for await (const event of runPlainChatResponse({
      request,
      llm,
      incomingText,
      ...(signal ? { signal } : {}),
      ...(trace ? { trace } : {}),
    })) {
      yield event;
    }
    runEndReason = "plain_chat";
    return;
  }

  const PlanningState = Annotation.Root({
    threadId: Annotation<string>(),
    locale: Annotation<string>(),
    researchTopic: Annotation<string>(),
    messages: Annotation<WorkflowState["messages"]>(),
    resources: Annotation<WorkflowState["resources"]>(),
    observations: Annotation<WorkflowState["observations"]>(),
    planIterations: Annotation<number>(),
    currentPlan: Annotation<WorkflowState["currentPlan"]>(),
    backgroundInvestigationResults: Annotation<WorkflowState["backgroundInvestigationResults"]>(),
    enableBackgroundInvestigation: Annotation<boolean>(),
    enableWebSearch: Annotation<boolean>(),
    maxPlanIterations: Annotation<number>(),
    maxStepNum: Annotation<number>(),
    maxSearchResults: Annotation<number>(),
    autoAcceptedPlan: Annotation<boolean>(),
    reportStyle: Annotation<WorkflowState["reportStyle"]>(),
    activeSkills: Annotation<WorkflowState["activeSkills"]>(),
    skillSelectionReason: Annotation<WorkflowState["skillSelectionReason"]>(),
    incomingText: Annotation<string>(),
    interruptFeedback: Annotation<ChatRequest["interrupt_feedback"]>(),
    isFeedback: Annotation<boolean>(),
    coordinatorAction: Annotation<PlanningGraphState["coordinatorAction"]>(),
    plannerShouldInterrupt: Annotation<boolean>(),
    done: Annotation<PlanningGraphState["done"]>(),
  });

  const recordChatEventAsTrace = (event: ChatEvent) => {
    const data = event.data;
    const agent = typeof data.agent === "string" ? data.agent : undefined;
    if (event.type === "tool_calls" && Array.isArray(data.tool_calls)) {
      for (const toolCall of data.tool_calls) {
        if (!toolCall || typeof toolCall !== "object") continue;
        const tc = toolCall as ToolCall;
        trace?.toolCallStarted({
          ...(currentSpanId ? { spanId: currentSpanId } : {}),
          toolCallId: tc.id,
          toolName: tc.name,
          input: tc.args,
          ...(agent ? { agent } : {}),
        });
      }
      return;
    }

    if (event.type === "tool_call_result") {
      const toolCallId = typeof data.tool_call_id === "string" ? data.tool_call_id : undefined;
      if (toolCallId) {
        trace?.toolCallEnded({
          ...(currentSpanId ? { spanId: currentSpanId } : {}),
          toolCallId,
          output: summarizeToolResult(data.content),
          ...(agent ? { agent } : {}),
        });
      }
      return;
    }

    if (event.type === "message_chunk") {
      trace?.message({
        ...(currentSpanId ? { spanId: currentSpanId } : {}),
        ...(agent ? { agent } : {}),
        metadata: {
          id: data.id,
          role: data.role,
          finish_reason: data.finish_reason,
          ...contentPreview(data.content),
          ...(typeof data.reasoning_content === "string"
            ? { reasoning_content_length: data.reasoning_content.length }
            : {}),
        },
      });
      return;
    }

    if (event.type === "interrupt") {
      trace?.interrupt({
        ...(currentSpanId ? { spanId: currentSpanId } : {}),
        ...(agent ? { agent } : {}),
        metadata: { id: data.id, options: data.options },
      });
    }
  };

  const emit = (config: LangGraphRunnableConfig | undefined, event: ChatEvent) => {
    recordChatEventAsTrace(event);
    config?.writer?.(event);
  };

  const coordinatorNode = async (s: PlanningGraphState, config?: LangGraphRunnableConfig) => tracedNode(
    trace,
    {
      spanId: `span_coordinator_${randomUUID()}`,
      name: "planning.coordinator",
      agent: "coordinator",
      input: { isFeedback: s.isFeedback, resources: s.resources.length },
      currentSpan: currentSpanId,
      setCurrentSpan: (spanId) => {
        currentSpanId = spanId;
      },
    },
    async () => {
    if (s.isFeedback) {
      return { coordinatorAction: "handoff_to_planner" as const };
    }

    if (s.resources.length > 0) {
      const coordinatorId = newMessageId();
      const content = isChineseLocale(s.locale)
        ? "收到。我会先制定一个学术研究计划，然后检索相关资料并撰写结构化研究报告。"
        : "Got it. I'll draft an academic research plan first, then retrieve evidence and write a structured research report.";

      emit(config, {
        type: "message_chunk",
        data: {
          thread_id: s.threadId,
          id: coordinatorId,
          agent: "coordinator",
          role: "assistant",
          content,
        },
      });
      emit(config, {
        type: "message_chunk",
        data: {
          thread_id: s.threadId,
          id: coordinatorId,
          agent: "coordinator",
          role: "assistant",
          finish_reason: "stop",
        },
      });

      return { coordinatorAction: "handoff_to_planner" as const };
    }

    const decision = await decideCoordinatorAction({
      llm,
      locale: s.locale,
      userText: s.incomingText,
      ...(config?.signal ? { signal: config.signal } : {}),
    });

    const coordinatorId = newMessageId();
    if (decision.action === "direct_response") {
      emit(config, {
        type: "message_chunk",
        data: {
          thread_id: s.threadId,
          id: coordinatorId,
          agent: "coordinator",
          role: "assistant",
          content: decision.message,
        },
      });
      emit(config, {
        type: "message_chunk",
        data: {
          thread_id: s.threadId,
          id: coordinatorId,
          agent: "coordinator",
          role: "assistant",
          finish_reason: "stop",
        },
      });
      return { coordinatorAction: "direct_response" as const, done: "direct_response" as const };
    }

    const content = isChineseLocale(s.locale)
      ? "收到。我会先制定一个学术研究计划，然后检索相关资料并撰写结构化研究报告。"
      : "Got it. I'll draft an academic research plan first, then retrieve evidence and write a structured research report.";

    emit(config, {
      type: "message_chunk",
      data: {
        thread_id: s.threadId,
        id: coordinatorId,
        agent: "coordinator",
        role: "assistant",
        content,
      },
    });
    emit(config, {
      type: "message_chunk",
      data: {
        thread_id: s.threadId,
        id: coordinatorId,
        agent: "coordinator",
        role: "assistant",
        finish_reason: "stop",
      },
    });

    return { coordinatorAction: "handoff_to_planner" as const };
    },
  );

  const backgroundInvestigatorNode = async (s: PlanningGraphState, config?: LangGraphRunnableConfig) => tracedNode(
    trace,
    {
      spanId: `span_background_investigator_${randomUUID()}`,
      name: "planning.background_investigator",
      agent: "researcher",
      input: { enabled: s.enableBackgroundInvestigation && s.enableWebSearch, query: s.researchTopic },
      currentSpan: currentSpanId,
      setCurrentSpan: (spanId) => {
        currentSpanId = spanId;
      },
    },
    async () => {
    const shouldPlan = !s.interruptFeedback || s.interruptFeedback === "edit_plan";
    if (!shouldPlan) return {};
    if (!s.enableBackgroundInvestigation || !s.enableWebSearch) return {};

    const academicToolCallId = newToolCallId();
    const webToolCallId = newToolCallId();
    const researcherId = newMessageId();
    const maxAcademicResults = academicSearchLimit(s);
    const toolCalls: ToolCall[] = [
      {
        type: "tool_call",
        id: academicToolCallId,
        name: "academic_search",
        args: { query: s.researchTopic, max_results: maxAcademicResults },
      },
      {
        type: "tool_call",
        id: webToolCallId,
        name: "web_search",
        args: { query: s.researchTopic, max_results: s.maxSearchResults },
      },
    ];
    const toolCallChunks: ToolCallChunk[] = toolCalls.map((toolCall, index) => ({
      type: "tool_call_chunk",
      index,
      id: toolCall.id,
      name: toolCall.name,
      args: JSON.stringify(toolCall.args),
    }));

    emit(config, {
      type: "tool_calls",
      data: {
        thread_id: s.threadId,
        id: researcherId,
        agent: "researcher",
        role: "assistant",
        tool_calls: toolCalls,
        tool_call_chunks: toolCallChunks,
      },
    });
    emit(config, {
      type: "message_chunk",
      data: {
        thread_id: s.threadId,
        id: researcherId,
        agent: "researcher",
        role: "assistant",
        finish_reason: "tool_calls",
      },
    });

    let academicText = "";
    let academicToolResult = "[]";
    try {
      const results = await academicSearch({
        query: s.researchTopic,
        maxResults: maxAcademicResults,
        ...(config?.signal ? { signal: config.signal } : {}),
      });
      academicText = formatAcademicResults(results);
      academicToolResult = academicResultsToToolResult(results);
    } catch (e) {
      academicText = e instanceof Error ? e.message : String(e);
      academicToolResult = JSON.stringify([{ type: "page", title: academicText, url: "", content: "" }]);
    }

    emit(config, {
      type: "tool_call_result",
      data: {
        thread_id: s.threadId,
        id: researcherId,
        agent: "researcher",
        role: "tool",
        tool_call_id: academicToolCallId,
        content: academicToolResult,
      },
    });

    let webText = "";
    let webToolResult = "[]";
    try {
      const results = await webSearch({
        query: s.researchTopic,
        maxResults: s.maxSearchResults,
        ...(config?.signal ? { signal: config.signal } : {}),
      });
      webText = results.length
        ? results
            .map((r, i) => `- [${i + 1}] ${r.title} (${r.url})${r.content ? `\n  ${r.content}` : ""}`)
            .join("\n")
        : "(no web search results)";
      webToolResult = JSON.stringify(
        results.map((r) => ({
          type: "page",
          title: r.title,
          url: r.url,
          content: r.content ?? "",
        })),
      );
    } catch (e) {
      webText = e instanceof Error ? e.message : String(e);
      webToolResult = JSON.stringify([{ type: "page", title: webText, url: "", content: "" }]);
    }

    emit(config, {
      type: "tool_call_result",
      data: {
        thread_id: s.threadId,
        id: researcherId,
        agent: "researcher",
        role: "tool",
        tool_call_id: webToolCallId,
        content: webToolResult,
      },
    });

    const investigationText = [`Academic literature search:\n${academicText}`, `Web search:\n${webText}`].join("\n\n");
    store.set({ ...(s as WorkflowState), backgroundInvestigationResults: investigationText });

    return { backgroundInvestigationResults: investigationText };
    },
  );

  const plannerNode = async (s: PlanningGraphState, config?: LangGraphRunnableConfig) => tracedNode(
    trace,
    {
      spanId: `span_planner_${randomUUID()}`,
      name: "planning.planner",
      agent: "planner",
      input: { isFeedback: s.isFeedback, interruptFeedback: s.interruptFeedback, planIterations: s.planIterations },
      currentSpan: currentSpanId,
      setCurrentSpan: (spanId) => {
        currentSpanId = spanId;
      },
    },
    async () => {
    const shouldPlan = !s.interruptFeedback || s.interruptFeedback === "edit_plan";
    if (!shouldPlan) {
      return { plannerShouldInterrupt: false, done: "none" as const };
    }

    let next: WorkflowState = s;

    const plannerId = newMessageId();
    const isEditing = s.interruptFeedback === "edit_plan";
    const plannerSkillContext = formatSkillPromptBlock(getSkills(next.activeSkills), "planner");
    const prompt =
      isEditing && next.currentPlan
        ? buildPlannerEditPrompt({
            locale: next.locale,
            query: next.researchTopic,
            currentPlan: next.currentPlan,
            instruction: s.incomingText,
            maxSteps: next.maxStepNum,
            enableWebSearch: next.enableWebSearch,
            backgroundInvestigationResults: next.backgroundInvestigationResults,
            skillContext: plannerSkillContext,
          })
        : buildPlannerPrompt({
            query: next.researchTopic,
            locale: next.locale,
            maxSteps: next.maxStepNum,
            enableWebSearch: next.enableWebSearch,
            backgroundInvestigationResults: next.backgroundInvestigationResults,
            skillContext: plannerSkillContext,
          });

    if (!llm) {
      const plan = buildFallbackPlan({
        query: next.researchTopic,
        maxSteps: next.maxStepNum,
        enableWebSearch: next.enableWebSearch,
      });
      const planText = JSON.stringify(plan, null, 2);
      emit(config, {
        type: "message_chunk",
        data: {
          thread_id: next.threadId,
          id: plannerId,
          agent: "planner",
          role: "assistant",
          content: planText,
        },
      });
      emit(config, {
        type: "message_chunk",
        data: {
          thread_id: next.threadId,
          id: plannerId,
          agent: "planner",
          role: "assistant",
          finish_reason: "stop",
        },
      });
      next = { ...next, currentPlan: plan, planIterations: next.planIterations + 1 };
      store.set(next);
    } else {
      let fullText = "";

      emit(config, {
        type: "message_chunk",
        data: {
          thread_id: next.threadId,
          id: plannerId,
          agent: "planner",
          role: "assistant",
          reasoning_content: "Planning…",
        },
      });

      try {
        for await (const delta of llm.streamChatCompletions({
          messages: [
            { role: "system", content: prompt.system },
            { role: "user", content: prompt.user },
          ],
          ...(config?.signal ? { signal: config.signal } : {}),
        })) {
          if (delta.reasoningContent) {
            emit(config, {
              type: "message_chunk",
              data: {
                thread_id: next.threadId,
                id: plannerId,
                agent: "planner",
                role: "assistant",
                reasoning_content: delta.reasoningContent,
              },
            });
          }
          if (delta.content) {
            fullText += stripThinkTags(delta.content);
          }
        }
      } catch (e) {
        const message = e instanceof Error ? e.message : String(e);
        emit(config, {
          type: "message_chunk",
          data: {
            thread_id: next.threadId,
            id: plannerId,
            agent: "planner",
            role: "assistant",
            reasoning_content: `Planner stream failed: ${message}`,
          },
        });
      }

      const parsedPlan = ensureTopicPlanFields({
        plan:
          safeParsePlan(fullText) ??
          buildFallbackPlan({
            query: next.researchTopic,
            maxSteps: next.maxStepNum,
            enableWebSearch: next.enableWebSearch,
          }),
        query: next.researchTopic,
        enableWebSearch: next.enableWebSearch,
      });

      const json = JSON.stringify(parsedPlan, null, 2);
      emit(config, {
        type: "message_chunk",
        data: {
          thread_id: next.threadId,
          id: plannerId,
          agent: "planner",
          role: "assistant",
          content: json,
        },
      });

      emit(config, {
        type: "message_chunk",
        data: {
          thread_id: next.threadId,
          id: plannerId,
          agent: "planner",
          role: "assistant",
          finish_reason: "stop",
        },
      });

      next = { ...next, currentPlan: parsedPlan, planIterations: next.planIterations + 1 };
      store.set(next);
    }

    const reachedMaxPlanIterations = next.planIterations >= next.maxPlanIterations;
    const shouldRun = next.autoAcceptedPlan || s.interruptFeedback === "accepted" || reachedMaxPlanIterations;
    if (!shouldRun) {
      return { plannerShouldInterrupt: true, done: "interrupt_ready" as const };
    }

    return {
      plannerShouldInterrupt: false,
      done: "none" as const,
      currentPlan: next.currentPlan,
      planIterations: next.planIterations,
      backgroundInvestigationResults: next.backgroundInvestigationResults,
    };
    },
  );

  const humanFeedbackNode = async (s: PlanningGraphState, config?: LangGraphRunnableConfig) => tracedNode(
    trace,
    {
      spanId: `span_human_feedback_${randomUUID()}`,
      name: "planning.human_feedback",
      agent: "planner",
      input: { plannerShouldInterrupt: s.plannerShouldInterrupt },
      currentSpan: currentSpanId,
      setCurrentSpan: (spanId) => {
        currentSpanId = spanId;
      },
    },
    async () => {
    if (!s.plannerShouldInterrupt) return {};
    const interruptId = `human_feedback:${randomUUID()}`;
    emit(config, {
      type: "interrupt",
      data: {
        thread_id: s.threadId,
        id: interruptId,
        agent: "planner",
        role: "assistant",
        finish_reason: "interrupt",
        options: [
          { text: "Edit plan", value: "edit_plan" },
          { text: "Start research", value: "accepted" },
        ],
      },
    });
    return {};
    },
  );

  const planningGraph = new StateGraph(PlanningState)
    .addNode("coordinator", coordinatorNode)
    .addNode("background_investigator", backgroundInvestigatorNode)
    .addNode("planner", plannerNode)
    .addNode("human_feedback", humanFeedbackNode)
    .addEdge(START, "coordinator")
    .addConditionalEdges("coordinator", (s: PlanningGraphState) => {
      if (s.coordinatorAction === "direct_response") return END;
      const shouldPlan = !s.interruptFeedback || s.interruptFeedback === "edit_plan";
      if (shouldPlan && s.enableBackgroundInvestigation && s.enableWebSearch) return "background_investigator";
      return "planner";
    })
    .addEdge("background_investigator", "planner")
    .addConditionalEdges("planner", (s: PlanningGraphState) => {
      return s.plannerShouldInterrupt ? "human_feedback" : END;
    })
    .addEdge("human_feedback", END)
    .compile();

  let finalPlanningState: PlanningGraphState = {
    ...state,
    incomingText,
    interruptFeedback: request.interrupt_feedback,
    isFeedback,
    coordinatorAction: null,
    plannerShouldInterrupt: false,
    done: "none",
  };

  const stream = await planningGraph.stream(finalPlanningState, {
    streamMode: ["custom", "values"],
    ...(signal ? { signal } : {}),
  });

  for await (const chunk of stream) {
    if (Array.isArray(chunk) && chunk.length === 2) {
      const [mode, payload] = chunk;
      if (mode === "custom") {
        const event = payload as ChatEvent;
        yield event;
      }
      if (mode === "values") {
        finalPlanningState = payload as PlanningGraphState;
      }
    }
  }

  if (finalPlanningState.done === "direct_response") {
    runEndReason = "direct_response";
    return;
  }

  if (finalPlanningState.plannerShouldInterrupt && finalPlanningState.done === "interrupt_ready") {
    runEndReason = "interrupt_ready";
    return;
  }

  state = store.get(request.thread_id) ?? state;
  const plan = state.currentPlan
    ? state.currentPlan
    : buildFallbackPlan({
        query: state.researchTopic,
        maxSteps: state.maxStepNum,
        enableWebSearch: state.enableWebSearch,
      });

  if (!state.currentPlan) {
    state = { ...state, currentPlan: plan };
    store.set(state);
  }

  const researcherNode = async (s: PlanningGraphState, config?: LangGraphRunnableConfig) => tracedNode(
    trace,
    {
      spanId: `span_researcher_${randomUUID()}`,
      name: "execution.researcher",
      agent: "researcher",
      input: { query: s.researchTopic, resources: s.resources.length, enableWebSearch: s.enableWebSearch },
      currentSpan: currentSpanId,
      setCurrentSpan: (spanId) => {
        currentSpanId = spanId;
      },
    },
    async () => {
    const researcherId = newMessageId();
    const researcherSkillContext = formatSkillPromptBlock(getSkills(s.activeSkills), "researcher");
    const maxAcademicResults = academicSearchLimit(s);
    const retrieved = await retrieveResources({
      query: s.researchTopic,
      resources: s.resources,
      limit: Math.max(1, Math.min(10, s.maxSearchResults)),
    });

    const retrieveToolCallId = newToolCallId();
    const academicToolCallId = newToolCallId();
    const toolCalls: ToolCall[] = [
      {
        type: "tool_call",
        id: retrieveToolCallId,
        name: "retrieve_resources",
        args: { query: s.researchTopic, limit: retrieved.length },
      },
      ...(s.enableWebSearch
        ? [
            {
              type: "tool_call" as const,
              id: academicToolCallId,
              name: "academic_search",
              args: { query: s.researchTopic, max_results: maxAcademicResults },
            },
          ]
        : []),
    ];
    const toolCallChunks: ToolCallChunk[] = toolCalls.map((toolCall, index) => ({
      type: "tool_call_chunk",
      index,
      id: toolCall.id,
      name: toolCall.name,
      args: JSON.stringify(toolCall.args),
    }));

    emit(config, {
      type: "tool_calls",
      data: {
        thread_id: s.threadId,
        id: researcherId,
        agent: "researcher",
        role: "assistant",
        tool_calls: toolCalls,
        tool_call_chunks: toolCallChunks,
      },
    });
    emit(config, {
      type: "message_chunk",
      data: {
        thread_id: s.threadId,
        id: researcherId,
        agent: "researcher",
        role: "assistant",
        finish_reason: "tool_calls",
      },
    });

    const retrievalText = retrieved
      .map((r, i) => {
        const desc = r.description ? `\n  ${r.description}` : "";
        const excerpt = r.excerpt
          ? `\n  Content:\n${r.excerpt
              .split("\n")
              .map((line) => `  ${line}`)
              .join("\n")}`
          : "";
        return `- [${i + 1}] ${r.title} (${r.uri})${desc}${excerpt}`;
      })
      .join("\n");
    const retrievalToolResult = JSON.stringify(
      retrieved.map((r, i) => ({
        id: r.uri,
        title: r.title,
        content: r.excerpt ?? r.description ?? `Resource ${i + 1}`,
      })),
    );

    emit(config, {
      type: "tool_call_result",
      data: {
        thread_id: s.threadId,
        id: researcherId,
        agent: "researcher",
        role: "tool",
        tool_call_id: retrieveToolCallId,
        content: retrievalToolResult,
      },
    });

    let academicText = "";
    if (s.enableWebSearch) {
      try {
        const academicResults = await academicSearch({
          query: s.researchTopic,
          maxResults: maxAcademicResults,
          ...(config?.signal ? { signal: config.signal } : {}),
        });
        academicText = formatAcademicResults(academicResults);
        emit(config, {
          type: "tool_call_result",
          data: {
            thread_id: s.threadId,
            id: researcherId,
            agent: "researcher",
            role: "tool",
            tool_call_id: academicToolCallId,
            content: academicResultsToToolResult(academicResults),
          },
        });
      } catch (e) {
        academicText = e instanceof Error ? e.message : String(e);
        emit(config, {
          type: "tool_call_result",
          data: {
            thread_id: s.threadId,
            id: researcherId,
            agent: "researcher",
            role: "tool",
            tool_call_id: academicToolCallId,
            content: JSON.stringify([{ type: "page", title: academicText, url: "", content: "" }]),
          },
        });
      }
    }

    emit(config, {
      type: "message_chunk",
      data: {
        thread_id: s.threadId,
        id: researcherId,
        agent: "researcher",
        role: "assistant",
        finish_reason: "stop",
      },
    });

    const updates = [
      researcherSkillContext ? `Active research skill guidance:\n${researcherSkillContext}` : null,
      hasSkill(s, "academic-paper-review") && s.resources.length
        ? "Paper review evidence note: prioritize uploaded or selected paper content. If the available content is incomplete, the final review must state that limitation."
        : null,
      hasSkill(s, "deep-research")
        ? "Deep research evidence note: synthesize across multiple angles, including facts, examples, current trends, comparisons, and limitations when available."
        : null,
      s.backgroundInvestigationResults ? `Background investigation:\n${s.backgroundInvestigationResults}` : null,
      retrievalText ? `Retrieved resources:\n${retrievalText}` : null,
      academicText ? `Academic literature search:\n${academicText}` : null,
    ].filter((x): x is string => Boolean(x));

    const next = {
      ...(s as WorkflowState),
      observations: [...s.observations, ...updates],
    };
    store.set(next);

    return { observations: next.observations };
    },
  );

  const reporterNode = async (s: PlanningGraphState, config?: LangGraphRunnableConfig) => tracedNode(
    trace,
    {
      spanId: `span_reporter_${randomUUID()}`,
      name: "execution.reporter",
      agent: "reporter",
      input: { observations: s.observations.length, resources: s.resources.length, style: s.reportStyle },
      currentSpan: currentSpanId,
      setCurrentSpan: (spanId) => {
        currentSpanId = spanId;
      },
    },
    async () => {
    const reporterId = newMessageId();
    const sources = [
      ...s.resources.map((r) => ({ title: r.title, uri: r.uri })),
      ...extractObservationSources(s.observations),
    ];
    const style = s.reportStyle;
    const observations = s.observations;
    const planForReport =
      s.currentPlan ??
      buildFallbackPlan({
        query: s.researchTopic,
        maxSteps: s.maxStepNum,
        enableWebSearch: s.enableWebSearch,
      });

    if (!llm) {
      const fallback = [
        `# ${s.researchTopic || planForReport.title || "Report"}`,
        "",
        `Style: ${style ?? "default"}`,
        "",
        "## Plan",
        planForReport.steps.map((step, i) => `${i + 1}. ${step.title} — ${step.description}`).join("\n"),
        "",
        "## Notes",
        observations.length ? observations.join("\n\n") : "(none)",
        "",
        "## Answer",
        "LLM is not configured. Set BASIC_MODEL__MODEL and BASIC_MODEL__API_KEY to enable full academic report generation.",
      ].join("\n");

      emit(config, {
        type: "message_chunk",
        data: {
          thread_id: s.threadId,
          id: reporterId,
          agent: "reporter",
          role: "assistant",
          content: fallback,
        },
      });
      emit(config, {
        type: "message_chunk",
        data: {
          thread_id: s.threadId,
          id: reporterId,
          agent: "reporter",
          role: "assistant",
          finish_reason: "stop",
        },
      });
      return {};
    }

    const reportPrompt = buildReporterPrompt({
      query: s.researchTopic,
      locale: s.locale,
      style,
      plan: planForReport,
      observations,
      sources,
      skillContext: formatSkillPromptBlock(getSkills(s.activeSkills), "reporter"),
    });

    emit(config, {
      type: "message_chunk",
      data: {
        thread_id: s.threadId,
        id: reporterId,
        agent: "reporter",
        role: "assistant",
        reasoning_content: "Writing…",
      },
    });

    try {
      for await (const delta of llm.streamChatCompletions({
        messages: [
          { role: "system", content: reportPrompt.system },
          { role: "user", content: reportPrompt.user },
        ],
        ...(config?.signal ? { signal: config.signal } : {}),
      })) {
        if (delta.reasoningContent) {
          emit(config, {
            type: "message_chunk",
            data: {
              thread_id: s.threadId,
              id: reporterId,
              agent: "reporter",
              role: "assistant",
              reasoning_content: delta.reasoningContent,
            },
          });
        }
        if (delta.content) {
          const cleaned = stripThinkTags(delta.content);
          if (!cleaned) continue;
          emit(config, {
            type: "message_chunk",
            data: {
              thread_id: s.threadId,
              id: reporterId,
              agent: "reporter",
              role: "assistant",
              content: cleaned,
            },
          });
        }
      }
    } catch (e) {
      const message = e instanceof Error ? e.message : String(e);
      emit(config, {
        type: "message_chunk",
        data: {
          thread_id: s.threadId,
          id: reporterId,
          agent: "reporter",
          role: "assistant",
          content: `\n\n[reporter_error] ${message}`,
        },
      });
    }

    emit(config, {
      type: "message_chunk",
      data: {
        thread_id: s.threadId,
        id: reporterId,
        agent: "reporter",
        role: "assistant",
        finish_reason: "stop",
      },
    });

    return {};
    },
  );

  const executionGraph = new StateGraph(PlanningState)
    .addNode("researcher", researcherNode)
    .addNode("reporter", reporterNode)
    .addEdge(START, "researcher")
    .addEdge("researcher", "reporter")
    .addEdge("reporter", END)
    .compile();

  let execState: PlanningGraphState = {
    ...(finalPlanningState as PlanningGraphState),
    ...state,
    incomingText,
    interruptFeedback: request.interrupt_feedback,
    isFeedback,
    done: "none",
  };

  const execStream = await executionGraph.stream(execState, {
    streamMode: ["custom", "values"],
    ...(signal ? { signal } : {}),
  });

  for await (const chunk of execStream) {
    if (Array.isArray(chunk) && chunk.length === 2) {
      const [mode, payload] = chunk;
      if (mode === "custom") {
        const event = payload as ChatEvent;
        yield event;
      }
      if (mode === "values") {
        execState = payload as PlanningGraphState;
      }
    }
  }
  return;
  } catch (e) {
    runStatus = signal?.aborted ? "aborted" : "error";
    trace?.error(e, currentSpanId ? { spanId: currentSpanId } : {});
    throw e;
  } finally {
    if (signal?.aborted) runStatus = "aborted";
    trace?.runEnded(runStatus, { reason: runEndReason });
    await trace?.flush();
  }
}

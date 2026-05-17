import { z } from "zod";

export const ResourceSchema = z.object({
  uri: z.string(),
  title: z.string(),
  description: z.string().optional().default(""),
});

export type Resource = z.infer<typeof ResourceSchema>;

export const SkillIdSchema = z.enum(["systematic-literature-review", "academic-paper-review", "deep-research"]);

export type SkillId = z.infer<typeof SkillIdSchema>;

export const ContentItemSchema = z.object({
  type: z.string(),
  text: z.string().optional(),
  image_url: z.string().optional(),
});

export const ChatMessageSchema = z.object({
  role: z.string(),
  content: z.union([z.string(), z.array(ContentItemSchema)]),
});

export const ChatRequestSchema = z.object({
  messages: z.array(ChatMessageSchema).optional().default([]),
  resources: z.array(ResourceSchema).optional().default([]),
  debug: z.boolean().optional().default(false),
  thread_id: z.string().optional().default("__default__"),
  locale: z.string().optional().default("en-US"),
  max_plan_iterations: z.number().int().positive().optional().default(1),
  max_step_num: z.number().int().positive().optional().default(3),
  max_search_results: z.number().int().positive().optional().default(3),
  auto_accepted_plan: z.boolean().optional().default(false),
  workflow_mode: z.enum(["chat", "research"]).optional().default("chat"),
  interrupt_feedback: z.string().nullable().optional(),
  mcp_settings: z.record(z.string(), z.unknown()).nullable().optional(),
  enable_background_investigation: z.boolean().optional().default(true),
  enable_web_search: z.boolean().optional().default(true),
  report_style: z
    .enum([
      "academic",
      "popular_science",
      "news",
      "social_media",
      "strategic_investment",
    ])
    .optional()
    .default("academic"),
  enable_deep_thinking: z.boolean().optional().default(false),
  enable_skills: z.boolean().optional().default(true),
  selected_skills: z.array(SkillIdSchema).optional().default([]),
  enable_clarification: z.boolean().nullable().optional(),
  max_clarification_rounds: z.number().int().positive().nullable().optional(),
  interrupt_before_tools: z.array(z.string()).optional().default([]),
});

export type ChatRequest = z.infer<typeof ChatRequestSchema>;

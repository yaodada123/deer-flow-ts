// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { MagicWandIcon } from "@radix-ui/react-icons";
import { AnimatePresence, motion } from "framer-motion";
import { ArrowUp, FileText, Lightbulb, Paperclip, X } from "lucide-react";
import { useTranslations } from "next-intl";
import { useCallback, useRef, useState } from "react";
import { toast } from "sonner";

import { Detective } from "~/components/deer-flow/icons/detective";
import MessageInput, {
  type MessageInputRef,
} from "~/components/deer-flow/message-input";
import { ReportStyleDialog } from "~/components/deer-flow/report-style-dialog";
import { Tooltip } from "~/components/deer-flow/tooltip";
import { BorderBeam } from "~/components/magicui/border-beam";
import { Button } from "~/components/ui/button";
import { enhancePrompt } from "~/core/api";
import { useConfig } from "~/core/api/hooks";
import { resolveServiceURL } from "~/core/api/resolve-service-url";
import type { Option, Resource } from "~/core/messages";
import {
  setEnableDeepThinking,
  setEnableBackgroundInvestigation,
  setWorkflowMode,
  useSettingsStore,
} from "~/core/store";
import { cn } from "~/lib/utils";

export function InputBox({
  className,
  responding,
  feedback,
  onSend,
  onCancel,
  onRemoveFeedback,
}: {
  className?: string;
  size?: "large" | "normal";
  responding?: boolean;
  feedback?: { option: Option } | null;
  onSend?: (
    message: string,
    options?: {
      interruptFeedback?: string;
      resources?: Array<Resource>;
    },
  ) => void;
  onCancel?: () => void;
  onRemoveFeedback?: () => void;
}) {
  const t = useTranslations("chat.inputBox");
  const tCommon = useTranslations("common");
  const enableDeepThinking = useSettingsStore(
    (state) => state.general.enableDeepThinking,
  );
  const workflowMode = useSettingsStore((state) => state.general.workflowMode);
  const backgroundInvestigation = useSettingsStore(
    (state) => state.general.enableBackgroundInvestigation,
  );
  const researchModeEnabled = workflowMode === "research";
  const { config, loading } = useConfig();
  const reportStyle = useSettingsStore((state) => state.general.reportStyle);
  const containerRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<MessageInputRef>(null);
  const feedbackRef = useRef<HTMLDivElement>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);

  // Enhancement state
  const [isEnhancing, setIsEnhancing] = useState(false);
  const [isEnhanceAnimating, setIsEnhanceAnimating] = useState(false);
  const [currentPrompt, setCurrentPrompt] = useState("");
  const [uploadingResource, setUploadingResource] = useState(false);
  const [attachedResources, setAttachedResources] = useState<Resource[]>([]);

  const handleSendMessage = useCallback(
    (message: string, resources: Array<Resource>) => {
      if (responding) {
        onCancel?.();
      } else {
        if (message.trim() === "") {
          return;
        }
        if (onSend) {
          const mergedResources = [...attachedResources, ...resources].filter(
            (resource, index, all) => all.findIndex((item) => item.uri === resource.uri) === index,
          );
          onSend(message, {
            interruptFeedback: feedback?.option.value,
            resources: mergedResources,
          });
          setAttachedResources([]);
          onRemoveFeedback?.();
          // Clear enhancement animation after sending
          setIsEnhanceAnimating(false);
        }
      }
    },
    [responding, onCancel, onSend, feedback, onRemoveFeedback, attachedResources],
  );

  const handleUploadResource = useCallback(async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    if (file.size === 0) {
      event.target.value = "";
      return;
    }

    setUploadingResource(true);
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch(resolveServiceURL("rag/upload"), {
        method: "POST",
        body: formData,
      });
      if (!response.ok) {
        const error = (await response.json().catch(() => null)) as { detail?: string } | null;
        throw new Error(error?.detail ?? `Upload failed: ${response.status}`);
      }
      const resource = (await response.json()) as Resource;
      setAttachedResources((prev) =>
        [...prev, resource].filter(
          (item, index, all) => all.findIndex((candidate) => candidate.uri === item.uri) === index,
        ),
      );
    } catch (error) {
      const message = error instanceof Error ? error.message : String(error);
      console.error("Failed to upload resource:", error);
      toast.error(message);
    } finally {
      setUploadingResource(false);
      event.target.value = "";
    }
  }, []);

  const handleEnhancePrompt = useCallback(async () => {
    if (currentPrompt.trim() === "" || isEnhancing) {
      return;
    }

    setIsEnhancing(true);
    setIsEnhanceAnimating(true);

    try {
      const enhancedPrompt = await enhancePrompt({
        prompt: currentPrompt,
        report_style: reportStyle.toUpperCase(),
      });

      // Add a small delay for better UX
      await new Promise((resolve) => setTimeout(resolve, 500));

      // Update the input with the enhanced prompt with animation
      if (inputRef.current) {
        inputRef.current.setContent(enhancedPrompt);
        setCurrentPrompt(enhancedPrompt);
      }

      // Keep animation for a bit longer to show the effect
      setTimeout(() => {
        setIsEnhanceAnimating(false);
      }, 1000);
    } catch (error) {
      console.error("Failed to enhance prompt:", error);
      setIsEnhanceAnimating(false);
      // Could add toast notification here
    } finally {
      setIsEnhancing(false);
    }
  }, [currentPrompt, isEnhancing, reportStyle]);

  return (
    <div
      className={cn(
        "bg-card relative flex h-full w-full flex-col rounded-[24px] border",
        className,
      )}
      ref={containerRef}
    >
      <div className="w-full">
        <AnimatePresence>
          {feedback && (
            <motion.div
              ref={feedbackRef}
              className="bg-background border-brand absolute top-0 left-0 mt-2 ml-4 flex items-center justify-center gap-1 rounded-2xl border px-2 py-0.5"
              initial={{ opacity: 0, scale: 0 }}
              animate={{ opacity: 1, scale: 1 }}
              exit={{ opacity: 0, scale: 0 }}
              transition={{ duration: 0.2, ease: "easeInOut" }}
            >
              <div className="text-brand flex h-full w-full items-center justify-center text-sm opacity-90">
                {feedback.option.text}
              </div>
              <X
                className="cursor-pointer opacity-60"
                size={16}
                onClick={onRemoveFeedback}
              />
            </motion.div>
          )}
          {isEnhanceAnimating && (
            <motion.div
              className="pointer-events-none absolute inset-0 z-20"
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.3 }}
            >
              <div className="relative h-full w-full">
                {/* Sparkle effect overlay */}
                <motion.div
                  className="absolute inset-0 rounded-[24px] bg-gradient-to-r from-blue-500/10 via-purple-500/10 to-blue-500/10"
                  animate={{
                    background: [
                      "linear-gradient(45deg, rgba(59, 130, 246, 0.1), rgba(147, 51, 234, 0.1), rgba(59, 130, 246, 0.1))",
                      "linear-gradient(225deg, rgba(147, 51, 234, 0.1), rgba(59, 130, 246, 0.1), rgba(147, 51, 234, 0.1))",
                      "linear-gradient(45deg, rgba(59, 130, 246, 0.1), rgba(147, 51, 234, 0.1), rgba(59, 130, 246, 0.1))",
                    ],
                  }}
                  transition={{ duration: 2, repeat: Infinity }}
                />
                {/* Floating sparkles */}
                {[...Array(6)].map((_, i) => (
                  <motion.div
                    key={i}
                    className="absolute h-2 w-2 rounded-full bg-blue-400"
                    style={{
                      left: `${20 + i * 12}%`,
                      top: `${30 + (i % 2) * 40}%`,
                    }}
                    animate={{
                      y: [-10, -20, -10],
                      opacity: [0, 1, 0],
                      scale: [0.5, 1, 0.5],
                    }}
                    transition={{
                      duration: 1.5,
                      repeat: Infinity,
                      delay: i * 0.2,
                    }}
                  />
                ))}
              </div>
            </motion.div>
          )}
        </AnimatePresence>
        <MessageInput
          className={cn(
            "h-24 px-4 pt-5",
            feedback && "pt-9",
            isEnhanceAnimating && "transition-all duration-500",
          )}
          ref={inputRef}
          loading={loading}
          config={config}
          onEnter={handleSendMessage}
          onChange={setCurrentPrompt}
        />
      </div>
      {attachedResources.length > 0 && (
        <div className="flex flex-wrap gap-2 px-4 pt-2">
          {attachedResources.map((resource) => (
            <div
              key={resource.uri}
              className="bg-muted text-muted-foreground flex items-center gap-1 rounded-full px-2 py-1 text-xs"
            >
              <FileText className="h-3 w-3" />
              <span className="max-w-40 truncate">{resource.title}</span>
              <button
                type="button"
                className="hover:text-foreground"
                onClick={() => setAttachedResources((prev) => prev.filter((item) => item.uri !== resource.uri))}
              >
                <X className="h-3 w-3" />
              </button>
            </div>
          ))}
        </div>
      )}
      <div className="flex items-center px-4 py-2">
        <div className="flex grow gap-2">
          <Tooltip
            className="max-w-60"
            title={
              <div>
                <h3 className="mb-2 font-bold">
                  {t("researchModeTooltip.title", {
                    status: researchModeEnabled ? t("on") : t("off"),
                  })}
                </h3>
                <p>{t("researchModeTooltip.description")}</p>
              </div>
            }
          >
            <Button
              className={cn(
                "rounded-2xl",
                researchModeEnabled && "!border-brand !text-brand",
              )}
              variant="outline"
              onClick={() => {
                setWorkflowMode(researchModeEnabled ? "chat" : "research");
              }}
            >
              <FileText /> {t("researchMode")}
            </Button>
          </Tooltip>

          {config?.models?.reasoning && config.models.reasoning.length > 0 && (
            <Tooltip
              className="max-w-60"
              title={
                <div>
                  <h3 className="mb-2 font-bold">
                    {t("deepThinkingTooltip.title", {
                      status: enableDeepThinking ? t("on") : t("off"),
                    })}
                  </h3>
                  <p>
                    {t("deepThinkingTooltip.description", {
                      model: config.models.reasoning[0] ?? "",
                    })}
                  </p>
                </div>
              }
            >
              <Button
                className={cn(
                  "rounded-2xl",
                  enableDeepThinking && "!border-brand !text-brand",
                )}
                variant="outline"
                onClick={() => {
                  setEnableDeepThinking(!enableDeepThinking);
                }}
              >
                <Lightbulb /> {t("deepThinking")}
              </Button>
            </Tooltip>
          )}

          <Tooltip
            className="max-w-60"
            title={
              <div>
                <h3 className="mb-2 font-bold">
                  {t("investigationTooltip.title", {
                    status: backgroundInvestigation ? t("on") : t("off"),
                  })}
                </h3>
                <p>{t("investigationTooltip.description")}</p>
              </div>
            }
          >
            <Button
              className={cn(
                "rounded-2xl",
                backgroundInvestigation && researchModeEnabled && "!border-brand !text-brand",
                !researchModeEnabled && "opacity-60",
              )}
              variant="outline"
              disabled={!researchModeEnabled}
              onClick={() =>
                setEnableBackgroundInvestigation(!backgroundInvestigation)
              }
            >
              <Detective /> {t("investigation")}
            </Button>
          </Tooltip>
          <ReportStyleDialog />
        </div>
        <div className="flex shrink-0 items-center gap-2">
          <input
            ref={fileInputRef}
            type="file"
            accept=".md,.txt,.pdf,text/markdown,text/plain,application/pdf"
            className="sr-only"
            onChange={handleUploadResource}
            disabled={uploadingResource}
          />
          <Tooltip title={t("uploadResource")}>
            <Button
              variant="ghost"
              size="icon"
              className={cn("hover:bg-accent h-10 w-10", uploadingResource && "animate-pulse")}
              onClick={() => fileInputRef.current?.click()}
              disabled={uploadingResource || responding}
            >
              <Paperclip className="text-brand" />
            </Button>
          </Tooltip>
          <Tooltip title={t("enhancePrompt")}>
            <Button
              variant="ghost"
              size="icon"
              className={cn(
                "hover:bg-accent h-10 w-10",
                isEnhancing && "animate-pulse",
              )}
              onClick={handleEnhancePrompt}
              disabled={isEnhancing || currentPrompt.trim() === ""}
            >
              {isEnhancing ? (
                <div className="flex h-10 w-10 items-center justify-center">
                  <div className="bg-foreground h-3 w-3 animate-bounce rounded-full opacity-70" />
                </div>
              ) : (
                <MagicWandIcon className="text-brand" />
              )}
            </Button>
          </Tooltip>
          <Tooltip title={responding ? tCommon("stop") : tCommon("send")}>
            <Button
              variant="outline"
              size="icon"
              className={cn("h-10 w-10 rounded-full")}
              onClick={() => inputRef.current?.submit()}
            >
              {responding ? (
                <div className="flex h-10 w-10 items-center justify-center">
                  <div className="bg-foreground h-4 w-4 rounded-sm opacity-70" />
                </div>
              ) : (
                <ArrowUp />
              )}
            </Button>
          </Tooltip>
        </div>
      </div>
      {isEnhancing && (
        <>
          <BorderBeam
            duration={5}
            size={250}
            className="from-transparent via-red-500 to-transparent"
          />
          <BorderBeam
            duration={5}
            delay={3}
            size={250}
            className="from-transparent via-blue-500 to-transparent"
          />
        </>
      )}
    </div>
  );
}

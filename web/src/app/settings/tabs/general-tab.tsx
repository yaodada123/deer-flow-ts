// Copyright (c) 2025 Bytedance Ltd. and/or its affiliates
// SPDX-License-Identifier: MIT

import { zodResolver } from "@hookform/resolvers/zod";
import { Settings } from "lucide-react";
import { useTranslations } from "next-intl";
import { useEffect, useMemo } from "react";
import { useForm } from "react-hook-form";
import { z } from "zod";

import { Checkbox } from "~/components/ui/checkbox";
import {
  Form,
  FormControl,
  FormDescription,
  FormField,
  FormItem,
  FormLabel,
  FormMessage,
} from "~/components/ui/form";
import { Input } from "~/components/ui/input";
import { Label } from "~/components/ui/label";
import { Switch } from "~/components/ui/switch";
import type { AcademicSkillId, SettingsState } from "~/core/store";

import type { Tab } from "./types";

const ACADEMIC_SKILLS: Array<{ id: AcademicSkillId; label: string; description: string }> = [
  {
    id: "systematic-literature-review",
    label: "Systematic Literature Review",
    description: "文献综述、survey、annotated bibliography 和跨论文综合。",
  },
  {
    id: "academic-paper-review",
    label: "Academic Paper Review",
    description: "单篇论文、arXiv、PDF 或审稿式分析。",
  },
  {
    id: "deep-research",
    label: "Deep Research",
    description: "多角度调研、对比、解释和证据综合。",
  },
];

const generalFormSchema = z.object({
  autoAcceptedPlan: z.boolean(),
  enableClarification: z.boolean(),
  maxClarificationRounds: z.number().min(1, {
    message: "Max clarification rounds must be at least 1.",
  }),
  maxPlanIterations: z.number().min(1, {
    message: "Max plan iterations must be at least 1.",
  }),
  maxStepNum: z.number().min(1, {
    message: "Max step number must be at least 1.",
  }),
  maxSearchResults: z.number().min(1, {
    message: "Max search results must be at least 1.",
  }),
  // Others
  workflowMode: z.enum(["chat", "research"]),
  enableBackgroundInvestigation: z.boolean(),
  enableDeepThinking: z.boolean(),
  enableWebSearch: z.boolean(),
  reportStyle: z.enum(["academic", "popular_science", "news", "social_media","strategic_investment"]),
  enableSkills: z.boolean(),
  selectedSkills: z.array(z.enum(["systematic-literature-review", "academic-paper-review", "deep-research"])),
});

export const GeneralTab: Tab = ({
  settings,
  onChange,
}: {
  settings: SettingsState;
  onChange: (changes: Partial<SettingsState>) => void;
}) => {
  const t = useTranslations("settings.general");
  const generalSettings = useMemo(
    () => ({
      ...settings.general,
      enableSkills: settings.skills.enableSkills,
      selectedSkills: settings.skills.selectedSkills,
    }),
    [settings],
  );
  const form = useForm<z.infer<typeof generalFormSchema>>({
    resolver: zodResolver(generalFormSchema, undefined, undefined),
    defaultValues: generalSettings,
    mode: "all",
    reValidateMode: "onBlur",
  });

  const currentSettings = form.watch();
  useEffect(() => {
    let hasChanges = false;
    for (const key in currentSettings) {
      const currentValue = currentSettings[key as keyof typeof currentSettings];
      const savedValue = generalSettings[key as keyof typeof generalSettings];
      if (JSON.stringify(currentValue) !== JSON.stringify(savedValue)) {
        hasChanges = true;
        break;
      }
    }
    if (hasChanges) {
      const { enableSkills, selectedSkills, ...general } = currentSettings;
      onChange({ general, skills: { enableSkills, selectedSkills } });
    }
  }, [currentSettings, generalSettings, onChange]);

  return (
    <div className="flex flex-col gap-4">
      <header>
        <h1 className="text-lg font-medium">{t("title")}</h1>
      </header>
      <main>
        <Form {...form}>
          <form className="space-y-8">
            <FormField
              control={form.control}
              name="autoAcceptedPlan"
              render={({ field }) => (
                <FormItem>
                  <FormControl>
                    <div className="flex items-center gap-2">
                      <Switch
                        id="autoAcceptedPlan"
                        checked={field.value}
                        onCheckedChange={field.onChange}
                      />
                      <Label className="text-sm" htmlFor="autoAcceptedPlan">
                        {t("autoAcceptPlan")}
                      </Label>
                    </div>
                  </FormControl>
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="enableClarification"
              render={({ field }) => (
                <FormItem>
                  <FormControl>
                    <div className="flex items-center gap-2">
                      <Switch
                        id="enableClarification"
                        checked={field.value}
                        onCheckedChange={field.onChange}
                      />
                      <Label className="text-sm" htmlFor="enableClarification">
                        {t("enableClarification")} {field.value ? "(On)" : "(Off)"}
                      </Label>
                    </div>
                  </FormControl>
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="enableWebSearch"
              render={({ field }) => (
                <FormItem>
                  <FormControl>
                    <div className="flex items-center gap-2">
                      <Switch
                        id="enableWebSearch"
                        checked={field.value}
                        onCheckedChange={field.onChange}
                      />
                      <Label className="text-sm" htmlFor="enableWebSearch">
                        {t("enableWebSearch")}
                      </Label>
                    </div>
                  </FormControl>
                  <FormDescription>
                    {t("enableWebSearchDescription")}
                  </FormDescription>
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="enableSkills"
              render={({ field }) => (
                <FormItem>
                  <FormControl>
                    <div className="flex items-center gap-2">
                      <Switch
                        id="enableSkills"
                        checked={field.value}
                        onCheckedChange={field.onChange}
                      />
                      <Label className="text-sm" htmlFor="enableSkills">
                        Enable academic skills
                      </Label>
                    </div>
                  </FormControl>
                  <FormDescription>
                    Automatically route research requests to ScholarFlow skills migrated from DeerFlow.
                  </FormDescription>
                </FormItem>
              )}
            />
            {form.watch("enableSkills") && (
              <FormField
                control={form.control}
                name="selectedSkills"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>Academic skills</FormLabel>
                    <FormDescription>
                      Leave all unchecked for automatic selection, or choose skills to force into every request.
                    </FormDescription>
                    <div className="space-y-3 pt-1">
                      {ACADEMIC_SKILLS.map((skill) => {
                        const checked = field.value.includes(skill.id);
                        return (
                          <div key={skill.id} className="flex items-start gap-2">
                            <Checkbox
                              id={`skill-${skill.id}`}
                              checked={checked}
                              onCheckedChange={(value) => {
                                const selected = new Set(field.value);
                                if (value === true) {
                                  selected.add(skill.id);
                                } else {
                                  selected.delete(skill.id);
                                }
                                field.onChange(Array.from(selected));
                              }}
                            />
                            <div className="grid gap-1 leading-none">
                              <Label className="text-sm" htmlFor={`skill-${skill.id}`}>
                                {skill.label}
                              </Label>
                              <p className="text-muted-foreground text-xs">{skill.description}</p>
                            </div>
                          </div>
                        );
                      })}
                    </div>
                  </FormItem>
                )}
              />
            )}
            {form.watch("enableClarification") && (
              <FormField
                control={form.control}
                name="maxClarificationRounds"
                render={({ field }) => (
                  <FormItem>
                    <FormLabel>{t("maxClarificationRounds")}</FormLabel>
                    <FormControl>
                      <Input
                        className="w-60"
                        type="number"
                        defaultValue={field.value}
                        min={1}
                        onChange={(event) =>
                          field.onChange(parseInt(event.target.value || "1"))
                        }
                      />
                    </FormControl>
                    <FormDescription>
                      {t("maxClarificationRoundsDescription")}
                    </FormDescription>
                    <FormMessage />
                  </FormItem>
                )}
              />
            )}
            <FormField
              control={form.control}
              name="maxPlanIterations"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>{t("maxPlanIterations")}</FormLabel>
                  <FormControl>
                    <Input
                      className="w-60"
                      type="number"
                      defaultValue={field.value}
                      min={1}
                      onChange={(event) =>
                        field.onChange(parseInt(event.target.value || "0"))
                      }
                    />
                  </FormControl>
                  <FormDescription>
                    {t("maxPlanIterationsDescription")}
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="maxStepNum"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>{t("maxStepsOfPlan")}</FormLabel>
                  <FormControl>
                    <Input
                      className="w-60"
                      type="number"
                      defaultValue={field.value}
                      min={1}
                      onChange={(event) =>
                        field.onChange(parseInt(event.target.value || "0"))
                      }
                    />
                  </FormControl>
                  <FormDescription>{t("maxStepsDescription")}</FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />
            <FormField
              control={form.control}
              name="maxSearchResults"
              render={({ field }) => (
                <FormItem>
                  <FormLabel>{t("maxSearchResults")}</FormLabel>
                  <FormControl>
                    <Input
                      className="w-60"
                      type="number"
                      defaultValue={field.value}
                      min={1}
                      onChange={(event) =>
                        field.onChange(parseInt(event.target.value || "0"))
                      }
                    />
                  </FormControl>
                  <FormDescription>
                    {t("maxSearchResultsDescription")}
                  </FormDescription>
                  <FormMessage />
                </FormItem>
              )}
            />
          </form>
        </Form>
      </main>
    </div>
  );
};
GeneralTab.displayName = "General";
GeneralTab.icon = Settings;

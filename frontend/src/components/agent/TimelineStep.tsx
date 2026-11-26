import { Check, X, Loader2 } from "lucide-react";
import type { ToolStep } from "@/hooks/useAgentStream";
import { StepIcon, toolLabel } from "./stepIcons";
import { cn } from "@/lib/utils";

/** One tool call (or note) nested under an agent. */
export function TimelineStep({ step }: { step: ToolStep }) {
  if (step.tool === "note") {
    return (
      <li className="step-enter flex gap-2 py-1 text-xs text-warn">
        <StepIcon tool="note" className="mt-0.5 h-3.5 w-3.5 shrink-0" />
        <span>{step.detail}</span>
      </li>
    );
  }
  return (
    <li className="step-enter flex gap-2 py-1">
      <StepIcon
        tool={step.tool}
        className={cn(
          "mt-0.5 h-3.5 w-3.5 shrink-0",
          step.status === "running" && "pulse text-accent",
          step.status === "done" && "text-muted",
          step.status === "error" && "text-danger",
        )}
      />
      <div className="min-w-0 flex-1">
        <div className="flex items-center gap-1.5">
          <span className="truncate text-xs font-medium">{toolLabel(step.tool)}</span>
          {step.status === "running" && <Loader2 className="h-3 w-3 animate-spin text-muted" />}
          {step.status === "done" && <Check className="h-3 w-3 text-ok" />}
          {step.status === "error" && <X className="h-3 w-3 text-danger" />}
        </div>
        {step.detail && (
          <p className="line-clamp-2 text-[11px] text-muted">{step.detail}</p>
        )}
      </div>
    </li>
  );
}

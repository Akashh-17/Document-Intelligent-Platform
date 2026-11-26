import { Activity, Check, Loader2, X } from "lucide-react";
import type { AgentStep } from "@/hooks/useAgentStream";
import type { PlanInfo } from "@/lib/events";
import { cn } from "@/lib/utils";
import { AGENT_ICONS } from "./stepIcons";
import { TimelineStep } from "./TimelineStep";

const ROUTE_LABEL: Record<PlanInfo["route"], string> = {
  documents: "your documents",
  web: "the web",
  both: "documents + web",
  direct: "no research needed",
};

function AgentSection({ step, last }: { step: AgentStep; last: boolean }) {
  const Icon = AGENT_ICONS[step.agent];
  return (
    <li className="step-enter relative flex gap-3 pb-4">
      {!last && <span className="absolute left-[15px] top-8 h-[calc(100%-1.5rem)] w-px bg-border" />}
      <span
        className={cn(
          "flex h-8 w-8 shrink-0 items-center justify-center rounded-full border",
          step.status === "running" && "border-accent text-accent",
          step.status === "done" && "border-ok/40 bg-ok/10 text-ok",
          step.status === "error" && "border-danger/40 bg-danger/10 text-danger",
        )}
      >
        <Icon className={cn("h-4 w-4", step.status === "running" && "pulse")} />
      </span>
      <div className="min-w-0 flex-1 pt-1">
        <div className="flex items-center gap-2">
          <span className="truncate text-sm font-semibold">{step.label}</span>
          {step.status === "running" && <Loader2 className="h-3.5 w-3.5 animate-spin text-muted" />}
          {step.status === "done" && <Check className="h-3.5 w-3.5 text-ok" />}
          {step.status === "error" && <X className="h-3.5 w-3.5 text-danger" />}
        </div>
        {(step.summary || step.detail) && (
          <p className="mt-0.5 line-clamp-3 text-xs text-muted">{step.summary || step.detail}</p>
        )}
        {step.tools.length > 0 && (
          <ul className="mt-1.5 border-l border-border pl-2.5">
            {step.tools.map((t) => (
              <TimelineStep key={t.id} step={t} />
            ))}
          </ul>
        )}
      </div>
    </li>
  );
}

export function ActivityTimeline({
  agents,
  plan,
  streaming,
}: {
  agents: AgentStep[];
  plan?: PlanInfo;
  streaming: boolean;
}) {
  return (
    <section className="rounded-xl border border-border bg-surface p-4">
      <header className="mb-3 flex items-center gap-2">
        <Activity className="h-4 w-4 text-accent" />
        <h2 className="text-sm font-semibold">Agent activity</h2>
        {plan && (
          <span
            title={plan.rationale}
            className="ml-auto rounded-full bg-surface-2 px-2 py-0.5 text-[11px] text-muted"
          >
            {ROUTE_LABEL[plan.route]}
          </span>
        )}
      </header>

      {agents.length === 0 ? (
        <p className="text-xs text-muted">
          {streaming ? "Thinking…" : "The planner, research agents and fact verifier will appear here."}
        </p>
      ) : (
        <ol className="relative">
          {agents.map((step, i) => (
            <AgentSection key={step.agent} step={step} last={i === agents.length - 1} />
          ))}
        </ol>
      )}
    </section>
  );
}

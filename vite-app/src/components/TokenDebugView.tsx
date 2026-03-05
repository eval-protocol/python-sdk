import { useState } from "react";

interface TokenTurnTrace {
  step_index: number;
  prompt_ids: number[];
  completion_ids: number[];
  completion_logprobs?: number[];
  detokenized_tokens?: string[];
  prompt_len?: number;
  completion_len?: number;
  step_reward?: number;
  tool_call_parser?: string;
}

interface FullEpisode {
  token_ids: number[];
  mask: number[]; // 0=prompt, >0 = turn index (completion)
  logprobs: (number | null)[];
  detokenized_tokens: string[];
  num_turns: number;
}

interface TokenDebugViewProps {
  extra: Record<string, any>;
}

type ColorMode = "mask" | "logprobs";

const TURN_COLORS = [
  "rgba(253, 224, 71, 0.5)",
  "rgba(134, 239, 172, 0.5)",
  "rgba(147, 197, 253, 0.5)",
  "rgba(249, 168, 212, 0.5)",
  "rgba(196, 181, 253, 0.5)",
  "rgba(252, 165, 165, 0.5)",
  "rgba(253, 186, 116, 0.5)",
  "rgba(94, 234, 212, 0.5)",
];

function turnColor(turnIdx: number): string {
  if (turnIdx <= 0) return "rgba(209, 213, 219, 0.3)";
  return TURN_COLORS[(turnIdx - 1) % TURN_COLORS.length];
}

function logprobToColor(lp: number): string {
  // Smooth gradient: 0 → bright green, -10 → deep red
  const clamped = Math.max(-10, Math.min(0, lp));
  const t = (clamped + 10) / 10; // 1.0 = logprob 0, 0.0 = logprob -10
  // Interpolate hue: 0 (red) → 120 (green)
  const hue = t * 120;
  const sat = 75 + (1 - t) * 15;
  const light = 45 + t * 15;
  const alpha = 0.35 + (1 - t) * 0.35;
  return `hsla(${hue}, ${sat}%, ${light}%, ${alpha})`;
}

function displayToken(token: string): string {
  return token
    .replace(/ /g, "\u00B7")
    .replace(/\n/g, "\u21B5\n")
    .replace(/\t/g, "\u2192  ");
}

function EpisodeToken({
  token,
  tokenId,
  turnIdx,
  logprob,
  colorMode,
  showIds,
}: {
  token: string;
  tokenId: number;
  turnIdx: number;
  logprob: number | null;
  colorMode: ColorMode;
  showIds: boolean;
}) {
  const [hover, setHover] = useState(false);
  const isCompletion = turnIdx > 0;

  let bgColor: string;
  if (colorMode === "logprobs" && isCompletion && logprob !== null) {
    bgColor = logprobToColor(logprob);
  } else {
    bgColor = turnColor(turnIdx);
  }

  const display = displayToken(token) || "\u2205";

  return (
    <span
      className="relative inline font-mono text-xs cursor-default border-b"
      style={{
        backgroundColor: bgColor,
        borderColor: hover ? "#374151" : "transparent",
        borderBottomWidth: isCompletion ? "2px" : "1px",
        borderBottomStyle: isCompletion ? "solid" : "dotted",
        borderBottomColor: isCompletion
          ? hover
            ? "#374151"
            : turnColor(turnIdx).replace("0.5)", "0.9)")
          : "transparent",
      }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      {display}
      {hover && (
        <span
          className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-1 px-2 py-1 bg-gray-900 text-white text-[10px] rounded whitespace-nowrap pointer-events-none"
          style={{ minWidth: "100px" }}
        >
          {isCompletion ? `completion (turn ${turnIdx})` : "prompt (masked)"}
          {showIds && (
            <>
              <br />
              id: {tokenId}
            </>
          )}
          {logprob !== null && (
            <>
              <br />
              logprob: {logprob.toFixed(4)}
            </>
          )}
        </span>
      )}
    </span>
  );
}

function FullEpisodeView({
  episode,
  colorMode,
  showIds,
}: {
  episode: FullEpisode;
  colorMode: ColorMode;
  showIds: boolean;
}) {
  const { token_ids, mask, logprobs, detokenized_tokens } = episode;

  const promptCount = mask.filter((m) => m === 0).length;
  const completionCount = mask.filter((m) => m > 0).length;

  return (
    <div className="border border-gray-200 rounded">
      <div className="flex items-center gap-3 px-3 py-1.5 bg-gray-50 border-b border-gray-200 text-xs">
        <span className="font-semibold">Full Episode ({episode.num_turns} turns)</span>
        <span className="text-gray-500">
          {token_ids.length} tokens total
        </span>
        <span
          className="px-1.5 py-0.5 rounded text-[10px]"
          style={{ backgroundColor: "rgba(209, 213, 219, 0.5)" }}
        >
          prompt (masked): {promptCount}
        </span>
        <span
          className="px-1.5 py-0.5 rounded text-[10px]"
          style={{ backgroundColor: TURN_COLORS[0] }}
        >
          completion (unmasked): {completionCount}
        </span>
      </div>

      {showIds && (
        <div className="px-3 py-2 border-b border-gray-100 overflow-x-auto">
          <div className="text-[10px] text-gray-500 mb-1 font-semibold uppercase tracking-wider">
            Token IDs (gray=masked/prompt, colored=unmasked/completion by turn) — hover for text &amp; logprob
          </div>
          <div className="flex flex-wrap gap-0.5">
            {token_ids.map((id, i) => (
              <TokenIdChip
                key={i}
                id={id}
                token={detokenized_tokens[i] ?? ""}
                turnIdx={mask[i]}
                logprob={logprobs[i]}
                colorMode={colorMode}
              />
            ))}
          </div>
        </div>
      )}

      <div className="px-3 py-2 overflow-x-auto">
        <div className="text-[10px] text-gray-500 mb-1 font-semibold uppercase tracking-wider">
          {colorMode === "logprobs"
            ? "Tokens: gray=masked prompt, completions colored by logprob"
            : "Tokens: gray=masked prompt, colored=unmasked completion (by turn)"}
        </div>
        <div className="whitespace-pre-wrap break-all leading-relaxed">
          {detokenized_tokens.map((tok, i) => (
            <EpisodeToken
              key={i}
              token={tok}
              tokenId={token_ids[i]}
              turnIdx={mask[i]}
              logprob={logprobs[i]}
              colorMode={colorMode}
              showIds={showIds}
            />
          ))}
        </div>
      </div>
    </div>
  );
}

function TurnSection({
  trace,
  colorMode,
  showIds,
}: {
  trace: TokenTurnTrace;
  colorMode: ColorMode;
  showIds: boolean;
}) {
  const promptLen = trace.prompt_len ?? trace.prompt_ids.length;
  const completionLen = trace.completion_len ?? trace.completion_ids.length;
  const allIds = [...trace.prompt_ids, ...trace.completion_ids];
  const detokens = trace.detokenized_tokens ?? [];
  const logprobs = trace.completion_logprobs ?? [];

  return (
    <div className="border border-gray-200 rounded">
      <div className="flex items-center gap-3 px-3 py-1.5 bg-gray-50 border-b border-gray-200 text-xs">
        <span className="font-semibold">Turn {trace.step_index}</span>
        <span className="text-gray-500">
          prompt: {promptLen} | completion: {completionLen}
        </span>
        {trace.step_reward !== undefined && (
          <span
            className={`font-mono ${trace.step_reward > 0 ? "text-green-600" : trace.step_reward < 0 ? "text-red-600" : "text-gray-600"}`}
          >
            reward: {trace.step_reward}
          </span>
        )}
        {trace.tool_call_parser && (
          <span className="text-gray-400">parser: {trace.tool_call_parser}</span>
        )}
      </div>

      {showIds && (
        <div className="px-3 py-2 border-b border-gray-100 overflow-x-auto">
          <div className="text-[10px] text-gray-500 mb-1 font-semibold uppercase tracking-wider">
            Token IDs
          </div>
          <div className="flex flex-wrap gap-0.5">
            {allIds.map((id, i) => (
              <span
                key={i}
                className="inline-block px-1 py-0.5 text-[10px] font-mono rounded"
                style={{
                  backgroundColor:
                    i < promptLen
                      ? "rgba(209, 213, 219, 0.4)"
                      : turnColor(trace.step_index),
                }}
              >
                {id}
              </span>
            ))}
          </div>
        </div>
      )}

      <div className="px-3 py-2 overflow-x-auto">
        <div className="whitespace-pre-wrap break-all leading-relaxed">
          {(detokens.length > 0 ? detokens : allIds.map((id) => `[${id}]`)).map(
            (tok, i) => {
              const isPrompt = i < promptLen;
              const lpIdx = i - promptLen;
              const lp =
                !isPrompt && lpIdx >= 0 && lpIdx < logprobs.length
                  ? logprobs[lpIdx]
                  : null;
              return (
                <EpisodeToken
                  key={i}
                  token={tok}
                  tokenId={allIds[i]}
                  turnIdx={isPrompt ? 0 : trace.step_index}
                  logprob={lp}
                  colorMode={colorMode}
                  showIds={showIds}
                />
              );
            }
          )}
        </div>
      </div>
    </div>
  );
}

function LogprobLegend() {
  const nStops = 20;
  const gradientStops = Array.from({ length: nStops }, (_, i) => {
    const lp = -10 + (10 * i) / (nStops - 1);
    return logprobToColor(lp);
  });
  const gradient = `linear-gradient(to right, ${gradientStops.join(", ")})`;
  return (
    <div className="flex items-center gap-1.5 text-[10px] text-gray-500">
      <span>Logprob:</span>
      <span className="text-gray-400">-10</span>
      <div
        className="rounded"
        style={{ background: gradient, width: "80px", height: "12px" }}
      />
      <span className="text-gray-400">0</span>
    </div>
  );
}

function TurnLegend({ numTurns }: { numTurns: number }) {
  return (
    <div className="flex items-center gap-1 text-[10px] text-gray-500">
      <span
        className="px-1.5 py-0.5 rounded"
        style={{ backgroundColor: "rgba(209, 213, 219, 0.5)" }}
      >
        masked
      </span>
      {Array.from({ length: Math.min(numTurns, 8) }, (_, i) => (
        <span
          key={i}
          className="px-1.5 py-0.5 rounded"
          style={{ backgroundColor: TURN_COLORS[i] }}
        >
          t{i + 1}
        </span>
      ))}
    </div>
  );
}

function TokenIdChip({
  id,
  token,
  turnIdx,
  logprob,
  colorMode = "mask",
}: {
  id: number;
  token: string;
  turnIdx: number;
  logprob: number | null;
  colorMode?: ColorMode;
}) {
  const [hover, setHover] = useState(false);
  const display = token
    .replace(/\n/g, "\\n")
    .replace(/\t/g, "\\t");

  const isCompletion = turnIdx > 0;
  const bg =
    colorMode === "logprobs" && isCompletion && logprob != null
      ? logprobToColor(logprob)
      : turnColor(turnIdx);

  return (
    <span
      className="relative inline-block px-1 py-0.5 text-[10px] font-mono rounded cursor-default hover:ring-1 hover:ring-gray-500"
      style={{ backgroundColor: bg }}
      onMouseEnter={() => setHover(true)}
      onMouseLeave={() => setHover(false)}
    >
      {id}
      {hover && (
        <span
          className="absolute z-50 bottom-full left-1/2 -translate-x-1/2 mb-1 px-2 py-1 bg-gray-900 text-white text-[10px] rounded whitespace-nowrap pointer-events-none"
          style={{ minWidth: "60px" }}
        >
          &quot;{display}&quot;
          {logprob != null && (
            <>
              <br />
              logprob: {logprob.toFixed(4)}
            </>
          )}
        </span>
      )}
    </span>
  );
}

function TextMaskView({
  episode,
  showIds,
  colorMode,
}: {
  episode: FullEpisode;
  showIds: boolean;
  colorMode: ColorMode;
}) {
  const { token_ids, mask, logprobs, detokenized_tokens } = episode;

  const promptTokens = mask.filter((m) => m === 0).length;
  const completionTokens = mask.filter((m) => m > 0).length;

  function bgForToken(i: number): string {
    const turnIdx = mask[i] ?? 0;
    if (colorMode === "logprobs" && turnIdx > 0 && logprobs[i] != null) {
      return logprobToColor(logprobs[i]!);
    }
    return turnColor(turnIdx);
  }

  // For mask mode, group consecutive tokens with same mask for cleaner spans
  type Segment = { turnIdx: number; text: string; bg: string };
  const segments: Segment[] = [];
  for (let i = 0; i < detokenized_tokens.length; i++) {
    const turnIdx = mask[i] ?? 0;
    const tok = detokenized_tokens[i] ?? "";
    const bg = bgForToken(i);
    if (
      colorMode === "mask" &&
      segments.length > 0 &&
      segments[segments.length - 1].turnIdx === turnIdx
    ) {
      segments[segments.length - 1].text += tok;
    } else {
      segments.push({ turnIdx, text: tok, bg });
    }
  }

  return (
    <div className="border border-gray-200 rounded">
      <div className="flex items-center gap-3 px-3 py-1.5 bg-gray-50 border-b border-gray-200 text-xs">
        <span className="font-semibold">Text + Mask ({episode.num_turns} turns)</span>
        <span
          className="px-1.5 py-0.5 rounded text-[10px]"
          style={{ backgroundColor: "rgba(209, 213, 219, 0.5)" }}
        >
          masked: {promptTokens}
        </span>
        <span
          className="px-1.5 py-0.5 rounded text-[10px]"
          style={{ backgroundColor: TURN_COLORS[0] }}
        >
          unmasked: {completionTokens}
        </span>
      </div>

      {showIds && (
        <div className="px-3 py-2 border-b border-gray-100 overflow-x-auto">
          <div className="text-[10px] text-gray-500 mb-1 font-semibold uppercase tracking-wider">
            Token IDs (gray=masked, colored=unmasked by turn) — hover for text &amp; logprob
          </div>
          <div className="flex flex-wrap gap-0.5">
            {token_ids.map((id, i) => (
              <TokenIdChip
                key={i}
                id={id}
                token={detokenized_tokens[i] ?? ""}
                turnIdx={mask[i]}
                logprob={logprobs[i]}
                colorMode={colorMode}
              />
            ))}
          </div>
        </div>
      )}

      <div className="px-3 py-2 overflow-x-auto whitespace-pre-wrap break-words leading-relaxed font-mono text-xs">
        {segments.map((seg, i) => (
          <span
            key={i}
            style={{
              backgroundColor: seg.bg,
              borderBottom:
                seg.turnIdx > 0
                  ? `2px solid ${turnColor(seg.turnIdx).replace("0.5)", "0.9)")}`
                  : "none",
            }}
          >
            {seg.text}
          </span>
        ))}
      </div>
    </div>
  );
}

type ViewLevel = "text" | "episode" | "turns";

export const TokenDebugView = ({ extra }: TokenDebugViewProps) => {
  const [colorMode, setColorMode] = useState<ColorMode>("mask");
  const [showIds, setShowIds] = useState(false);
  const [viewLevel, setViewLevel] = useState<ViewLevel>("text");

  const fullEpisode: FullEpisode | null = extra?.full_episode ?? null;
  const tokenTurnTraces: TokenTurnTrace[] = extra?.token_turn_traces ?? [];

  if (!fullEpisode && tokenTurnTraces.length === 0) {
    return (
      <div className="text-xs text-gray-400 italic p-2">
        No token data available
      </div>
    );
  }

  const episodeReward = extra?.episode_reward;
  const stepRewards: number[] = extra?.step_rewards ?? [];
  const numTurns = fullEpisode?.num_turns ?? tokenTurnTraces.length;

  return (
    <div className="w-[700px] flex-shrink-0">
      {/* Header */}
      <div className="flex items-center justify-between mb-2 flex-wrap gap-1">
        <div className="flex items-center gap-2">
          <h4 className="font-semibold text-xs text-gray-700">Token Debug</h4>
          {episodeReward !== undefined && (
            <span
              className={`text-xs font-mono px-1.5 py-0.5 rounded ${
                episodeReward > 0
                  ? "bg-green-100 text-green-700"
                  : episodeReward < 0
                    ? "bg-red-100 text-red-700"
                    : "bg-gray-100 text-gray-700"
              }`}
            >
              reward: {episodeReward}
            </span>
          )}
          {stepRewards.length > 0 && (
            <span className="text-[10px] text-gray-400 font-mono">
              [{stepRewards.map((r) => r.toFixed(1)).join(", ")}]
            </span>
          )}
        </div>
        <div className="flex items-center gap-2">
          {colorMode === "logprobs" ? <LogprobLegend /> : <TurnLegend numTurns={numTurns} />}
          <label className="flex items-center gap-1 text-[10px] text-gray-500 cursor-pointer">
            <input
              type="checkbox"
              checked={showIds}
              onChange={(e) => setShowIds(e.target.checked)}
              className="w-3 h-3"
            />
            IDs
          </label>
          {(["mask", "logprobs"] as ColorMode[]).map((m) => (
            <button
              key={m}
              onClick={() => setColorMode(m)}
              className={`px-2 py-0.5 text-[10px] rounded border ${
                colorMode === m
                  ? "bg-gray-800 text-white border-gray-800"
                  : "bg-white text-gray-600 border-gray-300 hover:bg-gray-100"
              }`}
            >
              {m}
            </button>
          ))}
          <span className="text-gray-300">|</span>
          {(["text", "episode", "turns"] as ViewLevel[]).map((v) => (
            <button
              key={v}
              onClick={() => setViewLevel(v)}
              className={`px-2 py-0.5 text-[10px] rounded border ${
                viewLevel === v
                  ? "bg-gray-800 text-white border-gray-800"
                  : "bg-white text-gray-600 border-gray-300 hover:bg-gray-100"
              }`}
            >
              {v}
            </button>
          ))}
        </div>
      </div>

      {/* Content */}
      <div className="max-h-[600px] overflow-y-auto space-y-3">
        {viewLevel === "text" && fullEpisode ? (
          <TextMaskView episode={fullEpisode} showIds={showIds} colorMode={colorMode} />
        ) : viewLevel === "episode" && fullEpisode ? (
          <FullEpisodeView
            episode={fullEpisode}
            colorMode={colorMode}
            showIds={showIds}
          />
        ) : tokenTurnTraces.length > 0 ? (
          tokenTurnTraces.map((trace, i) => (
            <TurnSection
              key={i}
              trace={trace}
              colorMode={colorMode}
              showIds={showIds}
            />
          ))
        ) : (
          <div className="text-xs text-gray-400 italic p-2">
            No token data available for this view
          </div>
        )}
      </div>
    </div>
  );
};

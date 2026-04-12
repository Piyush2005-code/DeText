import { useState } from "react";
import { Languages, Send, Loader2, ChevronDown, Timer } from "lucide-react";
import {
  Select,
  SelectContent,
  SelectItem,
  SelectTrigger,
  SelectValue,
} from "@/components/ui/select";

const API_URL = "http://localhost:8000";

const ALGORITHMS = [
  { value: "Complement Naive Bayes", label: "Complement Naive Bayes" },
  { value: "Passive Aggressive", label: "Passive Aggressive" },
  { value: "Ridge Classifier", label: "Ridge Classifier" },
  { value: "SGD Classifier", label: "SGD Classifier" },
  { value: "Lang Detect", label: "Lang Detect (Complement NB)" },
  { value: "FastText", label: "FastText" },
  { value: "GlotLID", label: "GlotLID" },
  { value: "CLD3", label: "CLD3" },
  { value: "CharCNN (High-Cap)", label: "CharCNN (High-Cap)" },
];

const Index = () => {
  const [text, setText] = useState("");
  const [algo, setAlgo] = useState(ALGORITHMS[0].value);
  const [result, setResult] = useState<string | null>(null);
  const [latency, setLatency] = useState<number | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const handleDetect = async () => {
    if (!text.trim()) return;
    setLoading(true);
    setResult(null);
    setLatency(null);
    setError(null);

    try {
      const res = await fetch(API_URL, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ prompt: text, detection_algo: algo }),
      });
      const data = await res.json();
      setResult(data["detected language"]);
      if (data["execution_time"] != null) {
        setLatency(data["execution_time"] * 1000);
      }
    } catch {
      setError("Could not reach the backend. Is it running?");
    } finally {
      setLoading(false);
    }
  };

  return (
    <div className="min-h-screen flex flex-col items-center justify-center px-4 py-12">
      {/* Header */}
      <div className="flex items-center gap-3 mb-2">
        <Languages className="w-8 h-8 text-primary" />
        <h1 className="font-heading text-4xl md:text-5xl font-bold tracking-tight">
          De<span className="text-primary">Text</span>
        </h1>
      </div>
      <p className="text-muted-foreground mb-10 text-center max-w-md">
        Paste any text and let our ML model detect its language instantly.
      </p>

      {/* Card */}
      <div className="w-full max-w-2xl rounded-xl border bg-card p-6 md:p-8 shadow-lg shadow-primary/5">
        <textarea
          value={text}
          onChange={(e) => setText(e.target.value)}
          placeholder="Paste or type text in any language…"
          rows={6}
          className="w-full resize-none rounded-lg border bg-secondary/50 px-4 py-3 text-foreground placeholder:text-muted-foreground focus:outline-none focus:ring-2 focus:ring-primary/40 transition font-body"
        />

        {/* Algorithm selector + Detect button row */}
        <div className="mt-4 flex gap-3 items-stretch">
          <div className="flex flex-col gap-1 min-w-[220px]">
            <label className="text-xs font-heading font-medium text-muted-foreground uppercase tracking-wider px-0.5">
              Algorithm
            </label>
            <Select value={algo} onValueChange={setAlgo}>
              <SelectTrigger
                id="algo-select"
                className="h-full rounded-lg border bg-secondary/50 font-body text-foreground focus:ring-2 focus:ring-primary/40"
              >
                <SelectValue placeholder="Select algorithm" />
                <ChevronDown className="w-4 h-4 opacity-50 ml-1" />
              </SelectTrigger>
              <SelectContent>
                {ALGORITHMS.map((a) => (
                  <SelectItem key={a.value} value={a.value} className="font-body">
                    {a.label}
                  </SelectItem>
                ))}
              </SelectContent>
            </Select>
          </div>

          <button
            onClick={handleDetect}
            disabled={loading || !text.trim()}
            className="flex-1 flex items-center justify-center gap-2 rounded-lg bg-primary px-6 py-3 font-heading font-semibold text-primary-foreground transition hover:brightness-110 disabled:opacity-40 disabled:cursor-not-allowed mt-5"
          >
            {loading ? (
              <>
                <Loader2 className="w-5 h-5 animate-spin" />
                Processing…
              </>
            ) : (
              <>
                <Send className="w-5 h-5" />
                Detect Language
              </>
            )}
          </button>
        </div>

        {/* Processing overlay */}
        {loading && (
          <div className="mt-6 flex flex-col items-center gap-3 py-6">
            <div className="flex gap-1.5">
              {[0, 1, 2].map((i) => (
                <span
                  key={i}
                  className="w-2.5 h-2.5 rounded-full bg-primary animate-pulse-glow"
                  style={{ animationDelay: `${i * 0.3}s` }}
                />
              ))}
            </div>
            <p className="text-muted-foreground font-heading text-sm tracking-wide uppercase">
              Processing
            </p>
          </div>
        )}

        {/* Result */}
        {result && !loading && (
          <div className="mt-6 rounded-lg border border-primary/20 bg-primary/5 p-5 text-center">
            <p className="text-xs uppercase tracking-widest text-muted-foreground mb-1 font-heading">
              Detected Language
            </p>
            <p className="text-3xl font-heading font-bold text-primary">{result}</p>
            {latency != null && (
              <div className="mt-3 inline-flex items-center gap-1.5 rounded-full border border-primary/20 bg-background px-3 py-1">
                <Timer className="w-3.5 h-3.5 text-muted-foreground" />
                <span className="text-xs font-heading text-muted-foreground">
                  {latency.toFixed(2)} ms · {algo}
                </span>
              </div>
            )}
          </div>
        )}

        {/* Error */}
        {error && !loading && (
          <div className="mt-6 rounded-lg border border-destructive/30 bg-destructive/5 p-4 text-center">
            <p className="text-destructive text-sm">{error}</p>
          </div>
        )}
      </div>

      <p className="mt-8 text-xs text-muted-foreground">
        Powered by <span className="font-heading font-semibold text-primary">DeText</span> ML Engine
      </p>
    </div>
  );
};

export default Index;

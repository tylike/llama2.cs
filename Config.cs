using System.Runtime.CompilerServices;
using System.Runtime.InteropServices;
#pragma warning disable CA2014

namespace llama2.cs;

// Transformer and RunState structs, and related memory management
[StructLayout(LayoutKind.Sequential)]
public struct Config
{
    public RunState InitializeRunState()
    {
        Config cfg = this;
        return new RunState
        {
            x = new float[cfg.dim],
            xb = new float[cfg.dim],
            xb2 = new float[cfg.dim],
            hb = new float[cfg.hidden_dim],
            hb2 = new float[cfg.hidden_dim],
            q = new float[cfg.dim],
            k = new float[cfg.dim],
            v = new float[cfg.dim],
            att = new float[cfg.n_heads * cfg.seq_len],
            logits = new float[cfg.vocab_size],
            probindex = new ProbIndex[cfg.vocab_size],
            key_cache = new float[cfg.n_layers * cfg.seq_len * cfg.dim],
            value_cache = new float[cfg.n_layers * cfg.seq_len * cfg.dim]
        };
    }

    public int dim; // transformer dimension
    public int hidden_dim; // for ffn layers
    public int n_layers; // number of layers
    public int n_heads; // number of query heads
    public int n_kv_heads; // number of key/value heads (can be < query heads because of multiquery)
    public int vocab_size; // vocabulary size, usually 256 (byte-level)
    public int seq_len; // max sequence length
}

using System.Runtime.InteropServices;
#pragma warning disable CA2014

namespace llama2.cs;

[StructLayout(LayoutKind.Sequential)]
public struct TransformerWeights
{
    // token embedding table
    public float[] token_embedding_table; // (vocab_size, dim)

    // weights for rmsnorms
    public ArraySegment<float> rms_att_weight; // (layer, dim) rmsnorm weights

    public ArraySegment<float> rms_ffn_weight; // (layer, dim)

    // weights for matmuls
    public ArraySegment<float> wq; // (layer, dim, dim)
    public ArraySegment<float> wk; // (layer, dim, dim)
    public ArraySegment<float> wv; // (layer, dim, dim)

    public ArraySegment<float> wo; // (layer, dim, dim)

    // weights for ffn
    public ArraySegment<float> w1; // (layer, hidden_dim, dim)
    public ArraySegment<float> w2; // (layer, dim, hidden_dim)

    public ArraySegment<float> w3; // (layer, hidden_dim, dim)

    // final rmsnorm
    public float[] rms_final_weight; // (dim,)

    // freq_cis for RoPE relatively positional embeddings
    public float[] freq_cis_real; // (seq_len, head_size/2)

    public float[] freq_cis_imag; // (seq_len, head_size/2)

    // (optional) classifier weights for the logits, on the last layer
    public float[] wcls;
}

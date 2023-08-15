#pragma warning disable CA2014

namespace llama2.cs;

/// <summary>
///     Used in top-p sampling
/// </summary>
public struct ProbIndex
{
    public float Prob;
    public int Index;

}
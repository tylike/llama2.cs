using System.Diagnostics.CodeAnalysis;
using System.Text;
#pragma warning disable CA2014

namespace llama2.cs;
public static class Program
{
    public static void Main(string[] args)
    {
        var engine = new Engine();
        engine.Run();
    }
}

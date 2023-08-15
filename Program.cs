using System.Diagnostics.CodeAnalysis;
using System.Text;
#pragma warning disable CA2014

namespace llama2.cs;
class ModelFile
{
    public string File { get; set; }
    public string Vocab { get; set; }
}
public static class Program
{
    static ModelFile zh = new ModelFile { File = @"D:\ai.study\llama2.cs-main\stories15M-llama2-enzh.bin", Vocab = @"D:\ai.study\llama2.c-zh-main\tokenizers\llama2enzh\tokenizer.bin" };
    static ModelFile baichuan = new ModelFile { File = @"D:\ai.study\llama2.cs-main\stories15M-baichuan.bin", Vocab = @"D:\ai.study\llama2.c-zh-main\tokenizers\baichuan\tokenizer.bin" };

    public static void Main(string[] args)
    {
        var modelFile = baichuan;
        var engine = new Engine();
        engine.Setup(modelFile.File, modelFile.Vocab);
        engine.Run();
        Console.ReadKey();
    }
    static void ErrorUsage()
    {
        Console.WriteLine("Usage:   run <checkpoint> [options]");
        Console.WriteLine("Example: run model.bin -n 256 -i \"Once upon a time\"");
        Console.WriteLine("Options:");
        Console.WriteLine("  -t <float>  temperature, default 1.0");
        Console.WriteLine("  -p <float>  p value in top-p (nucleus) sampling. default 0.9, 0 = off");
        Console.WriteLine("  -s <int>    random seed, default time(NULL)");
        Console.WriteLine("  -n <int>    number of steps to run for, default 256. 0 = max_seq_len");
        Console.WriteLine("  -i <string> input prompt");
    }
}

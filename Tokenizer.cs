using System.Text;
using static System.Net.Mime.MediaTypeNames;
#pragma warning disable CA2014

namespace llama2.cs;

public class Tokenizer
{
    public string[] vocab { get; private set; }
    public float[] vocabScores { get; private set; }
    public int maxTokenLength { get; private set; }
    public Config config { get; private set; }
    string vocabPath = @"D:\ai.study\llama2.cs-main\tokenizer.bin";
    public Tokenizer(Config config, string vocabPath = null)
    {
        this.config = config;
        if (vocabPath != null)
        {
            this.vocabPath = vocabPath;
        }
    }
    public void LoadVocab()
    {
        vocab = new string[config.vocab_size];
        vocabScores = new float[config.vocab_size];
        maxTokenLength = 0;

        using (FileStream fs = new FileStream(vocabPath, FileMode.Open,
                   FileAccess.Read))
        using (BinaryReader reader = new BinaryReader(fs))
        {
            try
            {
                maxTokenLength = reader.ReadInt32();

                for (int i = 0; i < config.vocab_size; i++)
                {
                    vocabScores[i] = reader.ReadSingle();

                    int len = reader.ReadInt32();
                    Span<byte> buffer = stackalloc byte[len]; // stack allocate buffer, assumes len is small
                    _ = reader.Read(buffer);

                    vocab[i] = Encoding.UTF8.GetString(buffer);
                }
            }
            catch (EndOfStreamException)
            {
                throw new Exception("failed read tokenizer.bin");
            }
        }
    }

    public (int[]? promptTokens, int numPromptTokens) EncodePrompt(string prompt)
    {
        int[]? promptTokens = null;
        int numPromptTokens = 0;
        if (!string.IsNullOrEmpty(prompt))
        {
            promptTokens = new int[prompt.Length];
            BpeEncode(prompt, vocab, vocabScores, config.vocab_size, maxTokenLength, ref promptTokens,
                ref numPromptTokens);
        }
        return (promptTokens, numPromptTokens);
    }
    public string Decode(int token, int next)
    {
        // following BOS (1) token, sentencepiece decoder strips any leading whitespace (see PR #89)
        //在BOS（1）标记之后，句子段解码器去除任何前导空格（参见PR#89）
        var nxt = vocab[next];
        if (token == 1 && nxt[0] == ' ')
        {
            return nxt.TrimStart();
        }
        return nxt;
        //return token == 1 && vocab[next][0] == ' ' ? vocab[next].TrimStart() : vocab[next];
    }
    public string Decode(int token)
    {
        return vocab[token];
    }


    private static void BpeEncode(string text, string[] vocab, float[] vocabScores, int vocabSize, int maxTokenLength,
        ref int[] tokens, ref int nTokens)
    {
        int StrLookup(string str, string[] vocab, int vocabSize)
        {
            for (int i = 0; i < vocabSize; i++)
                if (str == vocab[i])
                    return i;
            return -1;
        }

        StringBuilder strBuffer = new StringBuilder(maxTokenLength * 2 + 1); // *2 for concat, +1 for null terminator

        // first encode every individual byte in the input string
        nTokens = 0; // the number of tokens
        foreach (char c in text)
        {
            strBuffer.Clear();
            strBuffer.Append(c);

            int id = StrLookup(strBuffer.ToString(), vocab, vocabSize);
            if (id == -1)
            {
                Console.Error.WriteLine("not good");
                throw new Exception("Encoding error");
            }

            tokens[nTokens] = id;
            nTokens++;
        }

        // merge the best consecutive pair each iteration, according to the scores in vocab_scores
        while (true)
        {
            float bestScore = float.MinValue;
            int bestId = -1;
            int bestIdx = -1;

            for (int i = 0; i < nTokens - 1; i++)
            {
                // check if we can merge the pair (tokens[i], tokens[i+1])
                strBuffer.Clear();
                strBuffer.Append(vocab[tokens[i]]);
                strBuffer.Append(vocab[tokens[i + 1]]);

                int id = StrLookup(strBuffer.ToString(), vocab, vocabSize);
                if (id != -1 && vocabScores[id] > bestScore)
                {
                    // this merge pair exists in vocab! record its score and position
                    bestScore = vocabScores[id];
                    bestId = id;
                    bestIdx = i;
                }
            }

            if (bestIdx == -1) break; // we couldn't find any more pairs to merge, so we're done

            // merge the consecutive pair (bestIdx, bestIdx+1) into new token bestId
            tokens[bestIdx] = bestId;
            // delete token at position bestIdx+1, shift the entire sequence back 1
            for (int i = bestIdx + 1; i < nTokens - 1; i++) tokens[i] = tokens[i + 1];
            nTokens--; // token length decreased
        }
    }
}

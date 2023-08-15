using System.Diagnostics;
#pragma warning disable CA2014

namespace llama2.cs;

public class Engine
{

    long _rngSeed;
    Tokenizer tokenizer;
    Model model;
    Config config;
    float temperature = 1.0f;
    float topp = 0.9f;
    int rngSeed = 1;
    int steps = 256;
    string checkpoint = @"D:\ai.study\llama2.cs-main\stories15M.bin";
    int configVocabSize;
    public void Setup()
    {
        SetSeed((uint)DateTime.UtcNow.Ticks);
        if (_rngSeed == 0)
        {
            Console.WriteLine("Cannot use seed=0 because of the rng alg used\n");
            return;
        }
        model = new Model();
        model.LoadModel(checkpoint);

        // read in the model.bin file
        config = model.config;
        configVocabSize = config.vocab_size;
        // right now we cannot run for more than config.seq_len steps
        if (steps <= 0 || steps > config.seq_len) steps = config.seq_len;
        tokenizer = new Tokenizer(config);
        tokenizer.LoadVocab();
    }

    public void Run(string prompt = "what's your name?")
    {
        // create and init the application RunState
        RunState state = config.InitializeRunState();
        // process the prompt, if any
        (int[]? promptTokens, int numPromptTokens) = tokenizer.EncodePrompt(prompt);
        // start the main loop
        int token = 1; // init with token 1 (=BOS), as done in Llama-2 sentencepiece tokenizer
        int pos = 0; // position in the sequence
        Stopwatch timer = new Stopwatch();
        timer.Start();
        while (pos < steps)
        {
            // forward the transformer to get logits for the next token
            //转发transformer以获取下一个令牌的logits
            Transformer(token, pos, state);

            // advance the state state machine推进状态机
            int next; // will store the next token in the sequence将存储序列中的下一个令牌
            #region 原样将输入内容输出
            if (pos < numPromptTokens)
            {
                //如果我们仍在处理输入提示，请强制使用下一个提示标记
                // if we are still processing the input prompt, force the next prompt token
                next = promptTokens![pos];
            }
            #endregion
            else
            {
                //对下一个令牌进行采样
                // sample the next token
                if (temperature == 0.0f)
                {
                    //贪婪argmax采样：取概率最高的令牌
                    // greedy argmax sampling: take the token with the highest probability
                    next = Argmax(state.logits);
                }
                else
                {
                    //将温度应用于logits
                    // apply the temperature to the logits
                    for (int q = 0; q < config.vocab_size; q++) state.logits[q] /= temperature;
                    //将softmax应用于logits以获得下一个令牌的概率
                    // apply softmax to the logits to get the probabilities for next token
                    Softmax(state.logits, 0, configVocabSize);
                    //我们从这个分布中采样以获得下一个令牌
                    // we sample from this distribution to get the next token
                    if (topp <= 0)
                        //从预测的概率分布中简单采样
                        // simply sample from the predicted probability distribution
                        next = Sample(state.logits);
                    else
                        //top-p（nucleus）采样，将最不可能的标记箝位为零
                        // top-p (nucleus) sampling, clamping the least likely tokens to zero
                        next = SampleTopp(state.logits, state.probindex);
                }
            }

            pos++;

            #region 结束退出
            //依赖于数据的终止条件：BOS（1）标记对序列进行定界
            // data-dependent terminating condition: the BOS (1) token delimits sequences
            if (next == 1) break;
            #endregion

            #region 输出结果
            string tokenStr = tokenizer.Decode(token, next);// token == 1 && tokenizer.vocab[next][0] == ' ' ? tokenizer.vocab[next].TrimStart() : tokenizer.vocab[next];
            Console.Write(tokenStr);
            token = next;
            #endregion
        }

        timer.Stop();
        Console.WriteLine();

        // report achieved tok/s (pos-1 because the timer starts after first iteration)
        if (pos > 1)
            Console.WriteLine(
                $"achieved tok/s: {(pos - 1) / timer.Elapsed.Seconds}, tokens : {pos - 1} time : {timer.Elapsed}");
    }

    // This method sets the seed for the RNG
    void SetSeed(long seed)
    {
        _rngSeed = seed;
    }

    int RandomU32()
    {
        // xorshift rng: https://en.wikipedia.org/wiki/Xorshift#xorshift.2A
        _rngSeed ^= _rngSeed >> 12;
        _rngSeed ^= _rngSeed << 25;
        _rngSeed ^= _rngSeed >> 27;
        return (int)((_rngSeed * 0x2545F4914F6CDD1DL) >> 32);
    }

    float RandomF32()
    {
        // random float32 in [0,1)
        return (RandomU32() >>> 8) / 16777216.0f;
    }

    private int Argmax(float[] probabilities)
    {
        int maxI = 0;
        float maxP = probabilities[0];
        for (int i = 1; i < configVocabSize; i++)
            if (probabilities[i] > maxP)
            {
                maxI = i;
                maxP = probabilities[i];
            }

        return maxI;
    }


    int Sample(float[] probabilities)
    {
        float r = RandomF32();
        float cdf = 0.0f;
        for (int i = 0; i < configVocabSize; i++)
        {
            cdf += probabilities[i];
            if (r < cdf) return i;
        }

        return configVocabSize - 1;
    }

    private static int Compare(ProbIndex a, ProbIndex b)
    {
        if (a.Prob > b.Prob) return -1;
        if (a.Prob < b.Prob) return 1;
        return 0;
    }

    int SampleTopp(float[] probabilities, ProbIndex[] probindex)
    {
        for (int i = 0; i < configVocabSize; i++)
        {
            probindex[i].Index = i;
            probindex[i].Prob = probabilities[i];
        }

        Array.Sort(probindex, Compare);

        float cumulativeProb = 0.0f;
        int lastIdx = 0;
        for (int i = 0; i < configVocabSize; i++)
        {
            cumulativeProb += probindex[i].Prob;
            if (cumulativeProb > topp)
            {
                lastIdx = i;
                break;
            }
        }

        float r = RandomF32() * cumulativeProb;
        float cdf = 0.0f;
        for (int i = 0; i <= lastIdx; i++)
        {
            cdf += probindex[i].Prob;
            if (r < cdf) return probindex[i].Index;
        }

        return probindex[lastIdx].Index;
    }


    private static void Accum(float[] a, float[] b, int size)
    {
        for (int i = 0; i < size; i++) a[i] += b[i];
    }

    private static void Rmsnorm(float[] o, float[] x, ArraySegment<float> weight, int size)
    {
        // calculate sum of squares
        float ss = 0.0f;
        for (int j = 0; j < size; j++) ss += x[j] * x[j];
        ss /= size;
        ss += 1e-5f;
        ss = 1.0f / MathF.Sqrt(ss);

        // normalize and scale
        for (int j = 0; j < size; j++) o[j] = weight[j] * (ss * x[j]);
    }

    private void Softmax(float[] x, int xOffset, int size)
    {
        // find max value (for numerical stability)
        float maxVal = x[0 + xOffset];
        for (int i = 1; i < size; i++)
            if (x[i + xOffset] > maxVal)
                maxVal = x[i + xOffset];
        // exp and sum
        float sum = 0.0f;
        for (int i = 0; i < size; i++)
        {
            x[i + xOffset] = (float)Math.Exp(x[i + xOffset] - maxVal);
            sum += x[i + xOffset];
        }

        // normalize
        for (int i = 0; i < size; i++) x[i + xOffset] /= sum;
    }

    private static void Matmul(float[] xout, float[] x, ArraySegment<float> w, int n, int d)
    {
        // W (d,n) @ x (n,) . xout (d,)
        Parallel.For(0, d, i =>
        {
            float val = 0.0f;
            for (int j = 0; j < n; j++) val += w[i * n + j] * x[j];
            xout[i] = val;
        });
    }


    private void Transformer(int token, int pos, RunState state)
    {
        // a few convenience variables
        var config = model.config;
        var w = model.weights;

        int dim = config.dim;
        int hiddenDim = config.hidden_dim;
        int headSize = dim / config.n_heads;

        // copy the token embedding into x
        Array.Copy(w.token_embedding_table, token * dim, state.x, 0, dim);


        // forward all the layers
        for (int l = 0; l < config.n_layers; l++)
        {
            // attention rmsnorm
            Rmsnorm(state.xb, state.x, w.rms_att_weight[(l * dim)..], dim);

            // qkv matmuls for this position
            Matmul(state.q, state.xb, w.wq[(l * dim * dim)..], dim, dim);
            Matmul(state.k, state.xb, w.wk[(l * dim * dim)..], dim, dim);
            Matmul(state.v, state.xb, w.wv[(l * dim * dim)..], dim, dim);

            // RoPE relative positional encoding: complex-valued rotate q and k by freq_cis in each head
            for (int i = 0; i < dim; i += 2)
            {
                float q0 = state.q[i];
                float q1 = state.q[i + 1];
                float k0 = state.k[i];
                float k1 = state.k[i + 1];
                float fcr = w.freq_cis_real[pos * headSize / 2 + i % headSize / 2];
                float fci = w.freq_cis_imag[pos * headSize / 2 + i % headSize / 2];
                state.q[i] = q0 * fcr - q1 * fci;
                state.q[i + 1] = q0 * fci + q1 * fcr;
                state.k[i] = k0 * fcr - k1 * fci;
                state.k[i + 1] = k0 * fci + k1 * fcr;
            }

            // save key,value at this time step (pos) to our kv cache
            int loff = l * config.seq_len * dim; // kv cache layer offset for convenience
            Array.Copy(state.k, 0, state.key_cache, loff + pos * dim, dim);
            Array.Copy(state.v, 0, state.value_cache, loff + pos * dim, dim);

            // multihead attention. iterate over all heads
            Parallel.For(0, config.n_heads, h =>
            {
                // get the query vector for this head
                int qOffset = h * headSize;

                // attention scores for this head
                int attOffset = h * config.seq_len;

                // iterate over all timesteps, including the current one
                for (int t = 0; t <= pos; t++)
                {
                    // get the key vector for this head and at this timestep
                    int keyCacheOffset = loff + t * dim + h * headSize;

                    // calculate the attention score as the dot product of q and k
                    float score = 0.0f;
                    for (int i = 0; i < headSize; i++)
                        score += state.q[i + qOffset] * state.key_cache[i + keyCacheOffset];

                    score /= (float)Math.Sqrt(headSize);

                    // save the score to the attention buffer
                    state.att[t + attOffset] = score;
                }


                // softmax the scores to get attention weights, from 0..pos inclusively
                Softmax(state.att, attOffset, pos + 1);

                // weighted sum of the values, store back into xb
                int xbOffset = h * headSize;
                for (int i = xbOffset; i < xbOffset + headSize; i++) state.xb[i] = 0f;

                for (int t = 0; t <= pos; t++)
                {
                    // get the value vector for this head and at this timestep
                    int vOffset = loff + t * dim + h * headSize;

                    // get the attention weight for this timestep
                    float a = state.att[t + attOffset];

                    // accumulate the weighted value into xb
                    for (int i = 0; i < headSize; i++)
                        state.xb[i + xbOffset] += a * state.value_cache[i + vOffset];
                }
            });

            // final matmul to get the output of the attention
            Matmul(state.xb2, state.xb, w.wo[(l * dim * dim)..], dim, dim);

            // residual connection back into x
            Accum(state.x, state.xb2, dim);

            // ffn rmsnorm
            Rmsnorm(state.xb, state.x, w.rms_ffn_weight[(l * dim)..], dim);

            // Now for FFN in PyTorch we have: self.w2(F.silu(self.w1(x)) * self.w3(x))
            // first calculate self.w1(x) and self.w3(x)
            Matmul(state.hb, state.xb, w.w1[(l * dim * hiddenDim)..], dim, hiddenDim);
            Matmul(state.hb2, state.xb, w.w3[(l * dim * hiddenDim)..], dim, hiddenDim);

            // F.silu; silu(x)=x*σ(x),where σ(x) is the logistic sigmoid
            for (int i = 0; i < hiddenDim; i++)
                state.hb[i] *= (1.0f / (1.0f + (float)Math.Exp(-state.hb[i])));

            // elementwise multiply with w3(x)
            for (int i = 0; i < hiddenDim; i++) state.hb[i] *= state.hb2[i];

            // final matmul to get the output of the ffn
            Matmul(state.xb, state.hb, w.w2[(l * dim * hiddenDim)..], hiddenDim, dim);

            // residual connection
            Accum(state.x, state.xb, dim);
        }

        // final rmsnorm
        Rmsnorm(state.x, state.x, w.rms_final_weight, dim);

        // classifier into logits
        Matmul(state.logits, state.x, w.wcls, config.dim, config.vocab_size);
    }


}

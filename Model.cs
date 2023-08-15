using System.IO.MemoryMappedFiles;
using System.Runtime.InteropServices;
#pragma warning disable CA2014

namespace llama2.cs;

public class Model
{
    public Config config { get; private set; }
    public TransformerWeights weights { get; private set; }
    public void LoadModel(string checkpoint)
    {
        try
        {
            using FileStream fileStream = new FileStream(checkpoint, FileMode.Open, FileAccess.Read);
            // Read in the config header
            byte[] configBytes = new byte[Marshal.SizeOf(typeof(Config))];
            if (fileStream.Read(configBytes, 0, configBytes.Length) != configBytes.Length) Environment.Exit(1);

            GCHandle handle = GCHandle.Alloc(configBytes, GCHandleType.Pinned);
            try
            {
                IntPtr pointer = handle.AddrOfPinnedObject();
                config = (Config)Marshal.PtrToStructure(pointer, typeof(Config))!;
            }
            finally
            {
                handle.Free();
            }

            // Negative vocab size is a hacky way of signaling unshared weights. Bit yikes.
            bool sharedWeights = config.vocab_size > 0;
            var cfg = config;
            cfg.vocab_size = Math.Abs(config.vocab_size);
            config = cfg;

            // Figure out the file size
            var fileSize = fileStream.Length; // size of the checkpoint file in bytes

            using var memoryMappedFile = MemoryMappedFile.CreateFromFile(fileStream, null, fileSize,
                MemoryMappedFileAccess.Read, HandleInheritability.None, false);
            long configSizeInBytes = Marshal.SizeOf(typeof(Config));
            using var accessor = memoryMappedFile.CreateViewAccessor(configSizeInBytes,
                fileSize - configSizeInBytes, MemoryMappedFileAccess.Read);
            weights = new TransformerWeights();

            CheckpointInitWeights(accessor, sharedWeights);
        }
        catch (FileNotFoundException)
        {
            throw new Exception($"Couldn't open file {checkpoint}");

        }
        catch (Exception e)
        {
            throw new Exception($"Couldn't read {checkpoint}: {e.Message}");

        }

    }

    private void CheckpointInitWeights(MemoryMappedViewAccessor accessor, bool sharedWeights)
    {
        long offset = 0;
        var w = weights;
        var p = config;
        w.token_embedding_table = ReadFloatArray(accessor, ref offset, p.vocab_size * p.dim);
        w.rms_att_weight = ReadFloatArray(accessor, ref offset, p.n_layers * p.dim);
        w.wq = ReadFloatArray(accessor, ref offset, p.n_layers * p.dim * p.dim);
        w.wk = ReadFloatArray(accessor, ref offset, p.n_layers * p.dim * p.dim);
        w.wv = ReadFloatArray(accessor, ref offset, p.n_layers * p.dim * p.dim);
        w.wo = ReadFloatArray(accessor, ref offset, p.n_layers * p.dim * p.dim);
        w.rms_ffn_weight = ReadFloatArray(accessor, ref offset, p.n_layers * p.dim);
        w.w1 = ReadFloatArray(accessor, ref offset, p.n_layers * p.dim * p.hidden_dim);
        w.w2 = ReadFloatArray(accessor, ref offset, p.n_layers * p.hidden_dim * p.dim);
        w.w3 = ReadFloatArray(accessor, ref offset, p.n_layers * p.dim * p.hidden_dim);
        w.rms_final_weight = ReadFloatArray(accessor, ref offset, p.dim);
        int headSize = p.dim / p.n_heads;
        w.freq_cis_real = ReadFloatArray(accessor, ref offset, p.seq_len * headSize / 2);
        w.freq_cis_imag = ReadFloatArray(accessor, ref offset, p.seq_len * headSize / 2);

        if (sharedWeights) w.wcls = w.token_embedding_table;
        weights = w;
    }

    private static float[] ReadFloatArray(MemoryMappedViewAccessor accessor, ref long offset, int size)
    {
        float[] array = new float[size];
        accessor.ReadArray(offset, array, 0, size);
        offset += sizeof(float) * (long)size;
        return array;
    }
}

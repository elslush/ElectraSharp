using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using System.Collections.Specialized;

namespace Electra.Vocabulary;

public class Vocab
{
    private const int FileBufferSize = 1024;

    private readonly OrderedDictionary vocab;

    private Vocab(OrderedDictionary vocab)
    {
        this.vocab = vocab.AsReadOnly();
        IgnoreIds = new HashSet<int?>(new int?[] { this["[SEP]"], this["[CLS]"], this["[MASK]"] });
    }

    public int? this[string key] => (int?)vocab[key];

    public IReadOnlySet<int?> IgnoreIds { get; }

    public static Vocab FromFile(string fileName)
    {
        using var fileStream = File.OpenRead(fileName);
        return FromFileStream(fileStream);
    }

    public static Task<Vocab> FromFileAsync(string fileName, CancellationToken token = default)
    {
        using var fileStream = File.OpenRead(fileName);
        return FromFileStreamAsync(fileStream, token: token);
    }

    public static Vocab FromFileStream(Stream fileStream, int bufferSize = FileBufferSize)
    {
        using var streamReader = new StreamReader(fileStream, Encoding.Unicode, true, bufferSize);

        OrderedDictionary vocab = new();

        string? line;
        int i = 0;
        while((line = streamReader.ReadLine()) != null)
        {
            vocab.Add(line, i);
            i++;
        }

        return new(vocab);
    }

    public static async Task<Vocab> FromFileStreamAsync(Stream fileStream, int bufferSize = FileBufferSize, CancellationToken token = default)
    {
        using var streamReader = new StreamReader(fileStream, Encoding.Unicode, true, bufferSize);

        OrderedDictionary vocab = new();

        string? line;
        int i = 0;
        while ((line = await streamReader.ReadLineAsync()) != null)
        {
            vocab.Add(line, i);
            i++;

            token.ThrowIfCancellationRequested();
        }

        return new(vocab);
    }

    public static Vocab FromEnumerable(IEnumerable<string> values, int? knownCapacity = null)
    {

        OrderedDictionary vocab = knownCapacity is null ? new() : new(knownCapacity.Value);
        int i = 0;
        foreach (var value in values)
        {
            vocab.Add(value, i);
            i++;
        } 

        return new(vocab);
    }

    public static async Task<Vocab> FromAsyncEnumerable(IAsyncEnumerable<string> values, int? knownCapacity = null, CancellationToken token = default)
    {
        OrderedDictionary vocab = knownCapacity is null ? new() : new(knownCapacity.Value);
        int i = 0;
        await foreach (var value in values)
        {
            vocab.Add(value, i);
            i++;

            token.ThrowIfCancellationRequested();
        }

        return new(vocab);
    }
}

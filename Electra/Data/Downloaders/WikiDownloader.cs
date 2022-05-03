using ICSharpCode.SharpZipLib.BZip2;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Net;
using System.Text;
using System.Threading.Tasks;

namespace Electra.Data.Downloaders;

public class WikiDownloader : IDownloader, IDisposable
{
    private const string WikiUrl = "https://dumps.wikimedia.org/enwiki/latest/enwiki-latest-pages-articles.xml.bz2",
        WikiFileName = "wikicorpus_en.xml.bz2";
    private readonly HttpClient httpClient;
    private readonly FileStreamOptions fileStreamOptions;

    public WikiDownloader()
    {
        httpClient = new();
        fileStreamOptions = new FileStreamOptions()
        {
            Options = FileOptions.Asynchronous,
            Access = FileAccess.Write,
            Mode = FileMode.CreateNew,
        };
    }

    public async Task Download(CancellationToken token = default)
    {
        using var stream = await httpClient.GetStreamAsync(WikiUrl, token);
        using var fileStream = new FileStream(WikiFileName, fileStreamOptions);
        BZip2.Decompress(stream, fileStream, false);
    }

    public void Dispose()
    {
        httpClient.Dispose();
        GC.SuppressFinalize(this);
    }
}

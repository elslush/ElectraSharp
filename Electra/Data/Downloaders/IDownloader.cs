using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Electra.Data.Downloaders;

public interface IDownloader
{
    public Task Download(CancellationToken token = default);
}

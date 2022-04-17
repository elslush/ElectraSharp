using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;

namespace Electra.Model;

public class BertInput
{
    public int[] InputIds { get; init; }

    public int[] InputMask { get; init; }

    public int[] SegmentIds { get; init; }
}

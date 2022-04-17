using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using static Tensorflow.Binding;

namespace Electra.Model.Activations;

public static class GeluWrapper
{
    private static readonly Tensor twoConst = tf.constant(2.0, TF_DataType.TF_FLOAT);

    public static Tensor Gelu(Tensor x, string? _ = null)
    {
        var cdf = 0.5 * (1.0 + tf.math.erf(x / tf.sqrt(twoConst)));
        return x * cdf;
    }
}

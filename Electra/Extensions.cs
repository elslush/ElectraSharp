using static Tensorflow.Binding;
using Tensorflow;
using Tensorflow.NumPy;

namespace Electra;

public static class Extensions
{
    public static Shape GetShape(this Tensor tensor)
    {
        if (isinstance(tensor, typeof(NDArray)))
            return tensor.numpy().shape;
        else
            return tensor.shape;
    }

    public static Tensor sequence_mask<T1, T2>(this tensorflow tensorflow, T1 lengthsInput, T2 maxLenInput = default!, TF_DataType dtype = TF_DataType.TF_BOOL, string name = "SequenceMask")
    {
        dtype = dtype.as_base_dtype();
        return tf_with(ops.name_scope(name, values: new { lengthsInput, maxLenInput }), scope =>
        {
            var lengths = ops.convert_to_tensor(lengthsInput);

            Tensor maxLen;
            if (maxLenInput is null)
            {
                maxLen = gen_math_ops._max(lengths, tensorflow._all_dimensions(lengths));
                maxLen = gen_math_ops.maximum(tensorflow.constant(0, maxLen.dtype), maxLen);
            }
            else
                maxLen = ops.convert_to_tensor(maxLenInput);

            var row_vector = gen_math_ops.range(tensorflow.constant(0, maxLen.dtype), maxLen, tensorflow.constant(1, maxLen.dtype));
            var matrix = gen_math_ops.cast(tensorflow.expand_dims(lengths, -1), maxLen.dtype);
            var result = row_vector < matrix;
            if (dtype == TF_DataType.DtInvalid || result.dtype.is_compatible_with(dtype))
                return result;
            else
                return gen_math_ops.cast(result, dtype);
        });
    }

    public static Tensor gather_nd<T1, T2>(this tensorflow tensorflow, T1 @params, T2 indices, string name = "BatchGatherND", int batch_dims = 0)
    {
        var result = tensorflow.Context.ExecuteOp("GatherNd", name, new ExecuteOpArgs(
                @params,
                indices).SetAttributes(new { batch_dims }));
        return result[0];
    }

    private static Tensor _all_dimensions(this tensorflow tensorflow, Tensor x)
    {
        if (x.GetShape().dims.Length > 0)
            return constant_op.constant(np.arange(x.GetShape().ndim), dtype: dtypes.int32);

        return gen_math_ops.range(tensorflow.constant(0), tensorflow.rank(x), tensorflow.constant(1));
    }

    private static Tensor _all_dimensions(this tensorflow tensorflow, SparseTensor x)
    {
        if (x.dense_shape.GetShape().IsFullyDefined)
            return constant_op.constant(np.arange(x.dense_shape.GetShape().dims[0]), dtype: dtypes.int32);

        return gen_math_ops.range(tensorflow.constant(0), tensorflow.rank(x), tensorflow.constant(1));
    }
}

//using System;
//using System.Collections.Generic;
//using System.Linq;
//using System.Text;
//using System.Threading.Tasks;
//using Tensorflow;
//using Tensorflow.NumPy;
//using static Tensorflow.Binding;
//using static Tensorflow.KerasApi;
//using Tensorflow.Keras.Layers;
//using Tensorflow.Keras;
//using Tensorflow.Keras.ArgsDefinition;
//using Electra.Model.Activations;
//using Electra.Model;
//using System.Collections.Specialized;
//using Electra.Vocabulary;

//namespace Electra.Pretraining;

//public class PretrainingModel
//{
//    public PretrainingModel(PretrainingConfig config, BertInput features, Vocab vocab, bool isTraining)
//    {
//        BertConfig bertConfig = new();

//        var maskedInputs = DynamicMasking(features, config, vocab, config.MaskProbability);
//    }

//    private static (Tensor, Tensor, Tensor, Tensor, Tensor) DynamicMasking(BertInput features, PretrainingConfig config, Vocab vocab, float maskProb, float proposalDistribution = 1.0f, Tensor? disallowFromMask = null, Tensor? alreadyMasked = null)
//    {
//        Tensor input = tf.convert_to_tensor(features.InputIds, TF_DataType.TF_INT32);

//        var N = config.MaxPredictionsPerSeq;
//        var inputShape = features.InputIds.GetShape().as_int_list();
//        var B = inputShape[0];
//        var L = inputShape[1];

//        var candidatesMask = GetCandidatesMask(features, vocab, disallowFromMask);

//        var numTokens = tf.cast(tf.reduce_sum(input, -1), tf.float32);
//        var numToPredict = tf.maximum(1, tf.minimum(N, tf.cast(tf.round(numTokens * maskProb), tf.int32)));
//        var maskedLmWeights = tf.cast(tf.sequence_mask(numToPredict, N), tf.float32);

//        if (alreadyMasked is not null)
//            maskedLmWeights *= (1 - alreadyMasked);

//        // Get a probability of masking each position in the sequence
//        var candidate_mask_float = tf.cast(candidatesMask, tf.float32);
//        var sample_prob = proposalDistribution * candidate_mask_float;
//        sample_prob /= tf.reduce_sum(sample_prob, axis: -1, keepdims: true);

//        // Sample the positions to mask out
//        sample_prob = tf.stop_gradient(sample_prob);
//        var sample_logits = tf.log(sample_prob);
//        var masked_lm_positions = tf.random.categorical(sample_logits, N, output_dtype: tf.int32);
//        masked_lm_positions *= tf.cast(maskedLmWeights, tf.int32);

//        // Get the ids of the masked-out tokens
//        var shift = tf.expand_dims(L * tf.range(B), -1);
//        var flat_positions = tf.reshape(masked_lm_positions + shift, new Shape(-1, 1));
//        var masked_lm_ids = tf.gather_nd(tf.reshape(input, new Shape(-1)), flat_positions);
//        masked_lm_ids = tf.reshape(masked_lm_ids, new Shape(B, -1));
//        masked_lm_ids *= tf.cast(maskedLmWeights, tf.int32);

//        // Update the input ids

//        var replace_with_mask_positions = masked_lm_positions * tf.cast(tf.less(tf.random.uniform(new Shape(B, N)), 0.85f), tf.int32);
//        var inputs_ids = ScatterUpdate(input, tf.fill(new Shape(B, N), vocab["[MASK]"]), replace_with_mask_positions);

//        return (input, tf.stop_gradient(inputs_ids), masked_lm_positions, masked_lm_ids, maskedLmWeights);
//    }

//    private static Tensor ScatterUpdate(Tensor sequence, Tensor updates, Tensor positions)
//    {
//        var shape = sequence.GetShape();
//        var B = shape[0];
//        var L = shape[1];
//        long D;

//        bool depth_dimension = len(shape) == 3;
//        if (!depth_dimension)
//        {
//            D =1;
//            sequence = tf.expand_dims(sequence, -1);
//        }
//        else
//            D = shape[2];

//        var N = positions.GetShape()[1];

//        var shift = tf.expand_dims(L * tf.range(B), -1);
//        var flat_positions = tf.reshape(positions + shift, new Shape(-1, 1));
//        var flat_updates = tf.reshape(updates, new Shape(-1, D));
//        updates = tf.scatter_nd(flat_positions, flat_updates, new Shape(B * L, D));
//        updates = tf.reshape(updates, new Shape(B, L, D));

//        var flat_updates_mask = tf.ones(new Shape(B * N), tf.int32);
//        var updates_mask = tf.scatter_nd(flat_positions, flat_updates_mask, new Shape(B * L));
//        updates_mask = tf.reshape(updates_mask, [B, L]);
//        var not_first_token = tf.concat(new Tensor[] { tf.zeros(new Shape(B, 1), tf.int32), tf.ones(new Shape(B, L - 1), tf.int32) }, -1);
//        updates_mask *= not_first_token;
//        var updates_mask_3d = tf.expand_dims(updates_mask, -1);

//        if (sequence.dtype == tf.float32)
//        {
//            updates_mask_3d = tf.cast(updates_mask_3d, tf.float32);
//            updates /= tf.maximum(1.0, updates_mask_3d);
//        }
//        else
//            updates = tf.math.floordiv(updates, tf.maximum(1, updates_mask_3d));

//        updates_mask = tf.minimum(updates_mask, 1);
//        updates_mask_3d = tf.minimum(updates_mask_3d, 1);

//        var updated_sequence = (((1 - updates_mask_3d) * sequence) + (updates_mask_3d * updates));

//        if (!depth_dimension)
//            updated_sequence = tf.squeeze(updated_sequence, -1);

//        return updated_sequence;
//    }

//    private static Tensor GetCandidatesMask(BertInput features, Vocab vocab, Tensor? disallowFromMask)
//    {
//        var candidatesMask = tf.ones_like(features.InputIds, tf.@bool);
//        foreach (var ignoreId in vocab.IgnoreIds)
//            candidatesMask = tf.bitwise.bitwise_and(candidatesMask, tf.not_equal(features.InputIds, ignoreId));

//        if (disallowFromMask is not null)
//            candidatesMask = tf.bitwise.bitwise_and(candidatesMask, tf.bitwise.invert(candidatesMask));

//        return candidatesMask;
//    }
//}

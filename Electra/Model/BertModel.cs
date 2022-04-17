using System;
using System.Collections.Generic;
using System.Linq;
using System.Text;
using System.Threading.Tasks;
using Tensorflow;
using Tensorflow.NumPy;
using static Tensorflow.Binding;
using static Tensorflow.KerasApi;
using Tensorflow.Keras;
using Tensorflow.Keras.Layers;
using Tensorflow.Keras.ArgsDefinition;
using Electra.Model.Activations;

namespace Electra.Model;

public class BertModel
{
    private readonly BertConfig bertConfig;
    private Tensor embeddingOutput;

    public BertModel(
        BertConfig bertConfig,
        bool isTraining,
        Tensor inputIds,
        Tensor inputMask,
        Tensor tokenTypeIds,
        bool useOneHotEmbeddings = true,
        string scope = "electra",
        int? embeddingSize = null,
        Tensor? inputEmbeddings = null,
        Tensor? inputReprs = null,
        bool updateEmbeddings = true,
        bool untiedEmbeddings = false,
        bool ltr = false,
        bool rtl = false
        )
    {
        if (isTraining)
            this.bertConfig = bertConfig;
        else
            this.bertConfig = bertConfig with { HiddenDropoutProb = 0f, AttentionProbsDropoutProb = 0f };

        if (inputReprs is null)
        {
            Tensor tokenEmbeddings;
            if (inputEmbeddings is null)
            {
                int embeddingSizeLocal = embeddingSize ?? bertConfig.HiddenSize;
                tokenEmbeddings = EmbeddingLookup(
                    inputIds,
                    bertConfig.VocabSize,
                    embeddingSizeLocal,
                    bertConfig.InitializerRange,
                    "word_embeddings",
                    useOneHotEmbeddings
                    );
            }
            else
                tokenEmbeddings = inputEmbeddings;

            tf_with(tf.variable_scope((untiedEmbeddings ? scope : "electra") + "/embeddings", reuse: true), scope =>
            {
                embeddingOutput = EmbeddingPostprocessor(
                    tokenEmbeddings,
                    tokenTypeIds,
                    bertConfig.TypeVocabSize,
                    "token_type_embeddings",
                    true,
                    "position_embeddings",
                    bertConfig.InitializerRange,
                    bertConfig.MaxPositionEmbeddings,
                    bertConfig.HiddenDropoutProb
                );
            });
        }
        else
            embeddingOutput = inputReprs;

        var inputShape = tokenTypeIds.GetShape().as_int_list();
        var batch_size = inputShape[0];
        var seq_length = inputShape[1];

        tf_with(tf.variable_scope(scope, default_name: "electra"), scope =>
        {
            if (embeddingOutput.shape[-1] != bertConfig.HiddenSize)
            {
                var dense = new Dense(new DenseArgs() { Units = bertConfig.HiddenSize, Name = "embeddings_project" });
                embeddingOutput = dense.Apply(embeddingOutput);

                tf_with(tf.variable_scope("encoder"), scope =>
                {
                    var attentionMask = CreateAttentionMaskFromInputMask(tokenTypeIds, inputMask);

                    // tf.linalg.band_part is not implemented, cannot add causal masking :(
                    //if (ltr || rtl)
                    //{
                    //    var causalMask = tf.ones(new Shape(seq_length, seq_length));
                    //    if (ltr)
                    //        causalMask = tf.linalg.band_part(causalMask, -1, 0);
                    //    else
                    //        causalMask = tf.linalg.band_part(causalMask, 0, -1);
                    //    attentionMask *= tf.expand_dims(causalMask, axis: 0);
                    //}

                    var (all_layer_outputs, attn_maps) = TransformerModel(
                        input_tensor: embeddingOutput,
                        attention_mask: attentionMask,
                        hidden_size: bertConfig.HiddenSize,
                        num_hidden_layers: bertConfig.NumHiddenLayers,
                        num_attention_heads: bertConfig.NumAttentionHeads,
                        intermediate_size: bertConfig.IntermediateSize,
                        intermediate_act_fn: bertConfig.HiddenAct,
                        hidden_dropout_prob: bertConfig.HiddenDropoutProb,
                        attention_probs_dropout_prob: bertConfig.AttentionProbsDropoutProb,
                        initializer_range: bertConfig.InitializerRange,
                        do_return_all_layers: true
                        );

                    var sequenceOutput = all_layer_outputs[-1];
                    var pooledOutput = sequenceOutput.slice(new Slice(":, 0"));
                });
            }
        });
    }

    private static (Tensor, Tensor) TransformerModel(
        Tensor input_tensor,
        Tensor attention_mask, 
        int hidden_size= 768,
        int num_hidden_layers= 12,
        int num_attention_heads= 12,
        int intermediate_size= 3072,
        Activation? intermediate_act_fn= null,
        float hidden_dropout_prob= 0.1f,
        float attention_probs_dropout_prob = 0.1f,
        float initializer_range = 0.02f,
        bool do_return_all_layers = false)
    {
        if (intermediate_act_fn is null)
            intermediate_act_fn = GeluWrapper.Gelu;

        var attention_head_size = hidden_size / num_attention_heads;
        var input_shape = input_tensor.GetShape().as_int_list();
        var batch_size = input_shape[0];
        var seq_length = input_shape[1];
        var input_width = input_shape[2];

        var prev_output = ReshapeToMatrix(input_tensor);

        Tensor[] attn_maps = new Tensor[num_hidden_layers];
        Tensor[] all_layer_outputs = new Tensor[num_hidden_layers];
        for (int layer_idx = 0; layer_idx  < num_hidden_layers; layer_idx ++)
        {
            tf_with(tf.variable_scope($"layer_{layer_idx}"), scope =>
            {
                tf_with(tf.variable_scope("attention"), scope =>
                {
                    Tensor? attention_output = tf_with(tf.variable_scope("self"), scope =>
                    {
                        var (attention_head, probs) = AttentionLayer(
                        from_tensor: prev_output,
                        to_tensor: prev_output,
                        attention_mask: attention_mask,
                        num_attention_heads: num_attention_heads,
                        size_per_head: attention_head_size,
                        attention_probs_dropout_prob: attention_probs_dropout_prob,
                        initializer_range: initializer_range,
                        do_return_2d_tensor: true,
                        batch_size: batch_size,
                        from_seq_length: seq_length,
                        to_seq_length: seq_length
                        );

                        Tensor? attention_output;
                        attention_output  = attention_head;
                        attn_maps[layer_idx] = probs;

                        return attention_output;
                    });

                    tf_with(tf.variable_scope("output"), scope =>
                    {
                        var attention_denseArgs = new DenseArgs() { Units = hidden_size, KernelInitializer = CreateInitializer(initializer_range) };
                        var attention_dens = new Dense(attention_denseArgs);
                        attention_output = attention_dens.Apply(attention_output);
                        attention_output = Dropout(attention_output, hidden_dropout_prob);
                        attention_output = LayerNorm(attention_output + prev_output);
                    });

                    Tensor? intermediate_output = tf_with(tf.variable_scope("intermediate"), scope =>
                    {
                        var intermediate_denseArgs = new DenseArgs() { Units = intermediate_size, Activation = intermediate_act_fn, KernelInitializer = CreateInitializer(initializer_range) };
                        var intermediate_dens = new Dense(intermediate_denseArgs);
                        return intermediate_dens.Apply(attention_output);
                    });

                    tf_with(tf.variable_scope("output"), scope =>
                    {
                        var attention_denseArgs = new DenseArgs() { Units = hidden_size, KernelInitializer = CreateInitializer(initializer_range) };
                        var attention_dens = new Dense(attention_denseArgs);
                        prev_output  = attention_dens.Apply(intermediate_output);
                        prev_output = Dropout(attention_output, hidden_dropout_prob);
                        prev_output = LayerNorm(prev_output + attention_output);
                        all_layer_outputs[layer_idx] = prev_output;
                    });
                });
            });
        }

        var attn_map = tf.stack(attn_maps, 0);
        if (do_return_all_layers)
            return (tf.stack(all_layer_outputs.Select(layer => ReshapeFromMatrix(layer, input_shape)), 0), attn_map);
        else
            return (ReshapeFromMatrix(prev_output, input_shape), attn_map);
    }

    private static Tensor ReshapeToMatrix(Tensor inputTensor)
    {
        var ndims = inputTensor.shape.ndim;
        if (ndims == 2)
            return inputTensor;

        var width = inputTensor.shape[-1];
        return tf.reshape(inputTensor, new Shape(-1, width));
    }

    private static Tensor ReshapeFromMatrix(Tensor output_tensor, Shape orig_shape_list)
    {
        if (len(orig_shape_list) == 2)
            return output_tensor;

        var output_shape = tf.shape(output_tensor).shape;

        var orig_dims = orig_shape_list.Slice(0, -1);
        var width = output_shape[-1];

        return tf.reshape(output_tensor, new Shape(orig_dims.Append(width).ToArray()));
    }

    private static (Tensor, Tensor) AttentionLayer(
        Tensor from_tensor,
        Tensor to_tensor,
        Tensor? attention_mask = null,
        int num_attention_heads= 1,
        int size_per_head= 512,
        Activation? query_act = null,
        Activation? key_act = null,
        Activation? value_act = null,
        float attention_probs_dropout_prob= 0.0f,
        float initializer_range= 0.02f,
        bool do_return_2d_tensor= false,
        int batch_size = 1,
        int from_seq_length= 1,
        int to_seq_length = 1)
    {
        var from_tensor_2d = ReshapeToMatrix(from_tensor);
        var to_tensor_2d = ReshapeToMatrix(to_tensor);

        // query layer = [B*F, N*H]
        var query_denseArgs = new DenseArgs() { Units = num_attention_heads * size_per_head, Name = "query", KernelInitializer = CreateInitializer(initializer_range) };
        if (query_act is not null)
            query_denseArgs.Activation = query_act;

        var query_dense = new Dense(query_denseArgs);
        var query_layer = query_dense.Apply(from_tensor_2d);

        // key layer = [B*T, N*H]
        var key_denseArgs = new DenseArgs() { Units = num_attention_heads * size_per_head, Name = "key", KernelInitializer = CreateInitializer(initializer_range) };
        if (key_act is not null)
            key_denseArgs.Activation = key_act;

        var key_dense = new Dense(key_denseArgs);
        var key_layer = key_dense.Apply(to_tensor_2d);

        // value layer = [B*T, N*H]
        var value_denseArgs = new DenseArgs() { Units = num_attention_heads * size_per_head, Name = "value", KernelInitializer = CreateInitializer(initializer_range) };
        if (value_act is not null)
            value_denseArgs.Activation = value_act;

        var value_dense = new Dense(value_denseArgs);
        var value_layer = key_dense.Apply(to_tensor_2d);

        // query layer = [B, N, F, H]
        query_layer = TransposeForScores(query_layer, batch_size, num_attention_heads, from_seq_length, size_per_head);

        // key layer = [B, N, T, H]
        key_layer = TransposeForScores(key_layer, batch_size, num_attention_heads, to_seq_length, size_per_head);

        var attention_scores = tf.matmul(query_layer, key_layer, transpose_b: true);
        attention_scores = tf.multiply(attention_scores, 1.0 / Math.Sqrt(size_per_head));

        if (attention_mask is not null)
        {
            attention_mask = tf.expand_dims(attention_mask, axis: 1);
            var adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0;
            attention_scores += adder;
        }

        var attention_probs = tf.nn.softmax(attention_scores);
        attention_probs = Dropout(attention_probs, attention_probs_dropout_prob);

        // value layer = [B, T, N, H]
        value_layer = tf.reshape(value_layer, new Shape(batch_size, to_seq_length, num_attention_heads, size_per_head));

        // value layer = [B, N, T, H]
        value_layer = tf.transpose(value_layer, new Axis(0, 2, 1, 3));

        // context layer = [B, N, F, H]
        var context_layer = tf.matmul(attention_probs, value_layer);

        // context layer = [B, F, N, H]
        context_layer = tf.transpose(context_layer, new Axis(0, 2, 1, 3));

        if (do_return_2d_tensor)
            context_layer = tf.reshape(context_layer, new Shape(batch_size * from_seq_length, num_attention_heads * size_per_head));
        else
            context_layer = tf.reshape(context_layer, new Shape(batch_size, from_seq_length, num_attention_heads * size_per_head));

        return (context_layer, attention_probs);
    }

    private static Tensor TransposeForScores(Tensor inputTensor, int batch_size, int num_attention_heads, int seq_length, int width)
    {
        var outputTensor = tf.reshape(inputTensor, new Shape(batch_size, num_attention_heads, seq_length, width));
        outputTensor = tf.transpose(outputTensor, new Axis(0, 2, 1, 3));
        return outputTensor;
    }

    private static Tensor CreateAttentionMaskFromInputMask(Tensor fromTensor, Tensor toMask)
    {
        var fromShape = fromTensor.GetShape();
        var batch_size = fromShape[0];
        var from_seq_length = fromShape[1];

        var toShape = toMask.GetShape();
        var to_seq_length = toShape[1];

        var to_mask = tf.cast(tf.reshape(toMask, new Shape(batch_size, 1, to_seq_length)), tf.float32);
        var broadcast_ones = tf.ones(shape: new Shape(batch_size, from_seq_length, 1), dtype: tf.float32);

        var mask = broadcast_ones * to_mask;

        return mask;
    }

    private static Tensor EmbeddingLookup(Tensor inputIds, int vocabSize, int embeddingSize = 128, float initializerRange = 0.02f, string wordEmbeddingName = "word_embeddings", bool useOneHotEmbeddings = false)
    {
        var originalDims = inputIds.shape.ndim;
        if (originalDims == 2)
            inputIds = tf.expand_dims(inputIds);

        var embeddingsTable = tf.compat.v1
            .get_variable(name: wordEmbeddingName, shape: new Shape(vocabSize, embeddingSize), initializer: CreateInitializer(initializerRange))
            .AsTensor();

        var input_shape = inputIds.GetShape();

        Tensor output;
        if (originalDims == 3)
        {
            tf.reshape(inputIds, new Shape(-1, input_shape[3]));
            output = tf.matmul(inputIds, embeddingsTable);
            output = tf.reshape(output, new Shape(input_shape[0], input_shape[1], embeddingSize));
        }
        else
        {
            if (useOneHotEmbeddings)
            {
                var flat_input_ids = tf.reshape(inputIds, new Shape(-1));
                var one_hot_input_ids = tf.one_hot(flat_input_ids, depth: vocabSize);
                output = tf.matmul(one_hot_input_ids, embeddingsTable);
            }
            else
                output = tf.nn.embedding_lookup(embeddingsTable, inputIds);
        }

        output = tf.reshape(output, new Shape(input_shape[0], input_shape[1], input_shape[2] * embeddingSize));

        return output;
    }

    private static Tensor EmbeddingPostprocessor(
        Tensor inputTensor,
        Tensor token_type_ids,
        int token_type_vocab_size = 16,
        string token_type_embedding_name = "token_type_embeddings",
        bool use_position_embeddings = true,
        string position_embedding_name= "position_embeddings",
        float initializer_range= 0.02f,
        int max_position_embeddings = 512,
        float dropout_prob = 0.1f)
    {
        var inputShape = inputTensor.GetShape().as_int_list();
        var batch_size = inputShape[0];
        var seq_length = inputShape[1];
        var width = inputShape[2];

        var output = inputTensor;

        var token_type_table = tf.compat.v1
            .get_variable(name: token_type_embedding_name, shape: new Shape(token_type_vocab_size, width), initializer: CreateInitializer(initializer_range))
            .AsTensor();

        var flat_token_type_ids = tf.reshape(token_type_ids, new Shape(-1));
        var one_hot_ids = tf.one_hot(flat_token_type_ids, depth: token_type_vocab_size);
        var token_type_embeddings = tf.matmul(one_hot_ids, token_type_table);
        token_type_embeddings = tf.reshape(token_type_embeddings, new Shape(batch_size, seq_length, width));
        output += token_type_embeddings;

        if (use_position_embeddings)
        {
            var fullPositionEmbeddings = tf.compat.v1
            .get_variable(name: position_embedding_name, shape: new Shape(max_position_embeddings, width), initializer: CreateInitializer(initializer_range))
            .AsTensor();

            var positionEmbeddings = tf.slice(fullPositionEmbeddings, new int[] { 0, 0 }, new int[] { seq_length, -1 });

            var numDims = output.shape.as_int_list().Length;

            List<int> positionBroadcastShape = new(numDims);
            for (int i = 0; i < numDims - 2; i++)
                positionBroadcastShape[i] = 1;
            positionBroadcastShape[numDims - 2] = seq_length;
            positionBroadcastShape[numDims - 1] = width;

            positionEmbeddings = tf.reshape(positionEmbeddings, new Shape(positionBroadcastShape.ToArray()));
            output += positionEmbeddings;
        }

        output = LayerNormAndDropout(output, dropout_prob);

        return output;
    }

    private static Tensor Dropout(Tensor inputTensor, float dropoutProb)
    {
        if (dropoutProb == 0.0)
            return inputTensor;

        return tf.nn.dropout(inputTensor, rate: 1.0f - dropoutProb);
    }

    private static Tensor LayerNorm(Tensor inputTensor, string? name = null)
    {
        var normLayer = new LayerNormalization(new LayerNormalizationArgs());
        return normLayer.Apply(inputTensor);
    }

    private static Tensor LayerNormAndDropout(Tensor inputTensor, float dropoutProb, string? name = null)
    {
        var output_tensor = LayerNorm(inputTensor, name);
        return Dropout(output_tensor, dropoutProb);
    }

    private static IInitializer CreateInitializer(float initializerRange = 0.02f)
    {
        return tf.truncated_normal_initializer(stddev: initializerRange);
    }
}

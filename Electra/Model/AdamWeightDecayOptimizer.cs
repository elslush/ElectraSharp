//using System;
//using System.Collections.Generic;
//using System.Linq;
//using System.Text;
//using System.Threading.Tasks;
//using Tensorflow;
//using Tensorflow.NumPy;
//using static Tensorflow.Binding;
//using static Tensorflow.KerasApi;
//using Tensorflow.Keras;
//using Tensorflow.Keras.Layers;
//using Tensorflow.Keras.ArgsDefinition;
//using System.Text.RegularExpressions;

//namespace Electra.Model;

//public readonly struct AdamWeightDecayOptimizer
//{
//    private readonly float learning_rate,
//               weight_decay_rate,
//               beta_1,
//               beta_2,
//               epsilon;
//    private readonly string name;
//    private readonly HashSet<string> exclude_from_weight_decay;

//    public static AdamWeightDecayOptimizer CreateOptimizer(
//        float loss, 
//        float learning_rate,
//        int num_train_steps, 
//        float weight_decay_rate= 0.0f, 
//        bool use_tpu= false,
//        int warmup_steps= 0, 
//        int warmup_proportion= 0, 
//        float lr_decay_power = 1.0f,
//        float layerwise_lr_decay_power= -1f, 
//        int n_transformer_layers = 0)
//    {
//        learning_rate = tf.train.polynomial_decay(
//      learning_rate,
//      global_step,
//      num_train_steps,
//      end_learning_rate=0.0,
//      power=lr_decay_power,
//      cycle=False)
//    }

//    private AdamWeightDecayOptimizer(float learning_rate,
//               float weight_decay_rate= 0.0f,
//               float beta_1 = 0.9f,
//               float beta_2 = 0.999f,
//               float epsilon = 1e-6f,
//               string name = nameof(AdamWeightDecayOptimizer),
//               params string[] exclude_from_weight_decay
//    )
//    {
//        this.learning_rate = learning_rate;
//        this.weight_decay_rate = weight_decay_rate;
//        this.beta_1 = beta_1;
//        this.beta_2 = beta_2;
//        this.epsilon = epsilon;
//        this.name = name;
//        this.exclude_from_weight_decay = exclude_from_weight_decay.ToHashSet();
//    }

//    private void _apply_gradients(grads_and_vars, learning_rate)
//    {
//    assignments = []
//    for (grad, param) in grads_and_vars:
//        if grad is None or param is None:
//        continue

//      param_name = self._get_variable_name(param.name)

//      m = tf.get_variable(
//          name=param_name + "/adam_m",
//          shape=param.shape.as_list(),
//          dtype=tf.float32,
//          trainable=False,
//          initializer=tf.zeros_initializer())
//      v = tf.get_variable(
//          name=param_name + "/adam_v",
//          shape=param.shape.as_list(),
//          dtype=tf.float32,
//          trainable=False,
//          initializer=tf.zeros_initializer())

//      # Standard Adam update.
//        next_m = (
//          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
//      next_v = (
//          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
//                                                    tf.square(grad)))
//      update = next_m / (tf.sqrt(next_v) + self.epsilon)

//        if self.weight_decay_rate > 0:
//        if self._do_use_weight_decay(param_name):
//          update += self.weight_decay_rate* param

//      update_with_lr = learning_rate* update
//      next_param = param - update_with_lr

//      assignments.extend(
//          [param.assign(next_param),
//           m.assign(next_m),
//           v.assign(next_v)])

//    return assignments
//    }

//    public void apply_gradients(self, grads_and_vars, global_step= None, name= None)
//    {
//        if isinstance(learning_rate, dict) :
//      key_to_grads_and_vars = { }
//        for grad, var in grads_and_vars:

//          update_for_var = False
//          for key in self.learning_rate:
//          if key in var.name:
//            update_for_var = True
//            if key not in key_to_grads_and_vars:
//        key_to_grads_and_vars[key] = []
//    key_to_grads_and_vars[key].append((grad, var))
//        if not update_for_var:
//            raise ValueError("No learning rate specified for variable", var)
//      assignments = []
//      for key, key_grads_and_vars in key_to_grads_and_vars.items():
//        assignments += self._apply_gradients(key_grads_and_vars,
//                                             self.learning_rate[key])
//    else:
//      assignments = self._apply_gradients(grads_and_vars, self.learning_rate)
//    return tf.group(*assignments, name=name)
//    }
    

//    private bool _do_use_weight_decay(string paramName)
//    {
//        if (weight_decay_rate <= 0)
//            return false;

//        return exclude_from_weight_decay.Contains(paramName);
//    }

//    private static readonly Regex VariableNameRegex = new("^(.*):\\d+$");
//    private string _get_variable_name(string paramName)
//    {
//        var match = VariableNameRegex.Match(paramName);
//        if (match.Success)
//            return match.Groups[1].Value;
//        return paramName;
//    }

//    private 
//def _get_layer_lrs(learning_rate, layer_decay, n_layers):
//  """Have lower learning rates for layers closer to the input."""
//  key_to_depths = collections.OrderedDict({
//        "/embeddings/": 0,
//      "/embeddings_project/": 0,
//      "task_specific/": n_layers + 2,
//  })
//  for layer in range(n_layers) :
//    key_to_depths["encoder/layer_" + str(layer) + "/"] = layer + 1
//  return {
//      key: learning_rate* (layer_decay** (n_layers + 2 - depth))
//      for key, depth in key_to_depths.items()
//}
//}

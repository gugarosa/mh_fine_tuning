# Improving Pre-Trained Weights Through Meta-Heuristic Fine-Tuning

*This repository holds all the necessary code to run the very-same experiments described in the paper "Improving Pre-Trained Weights Through Meta-Heuristic Fine-Tuning".*

---

## References

If you use our work to fulfill any of your needs, please cite us:

```
```

---

## Structure

 * `core`
   * `model.py`: Defines the base Machine Learning architecture;
 * `models`
   * `cnn.py`: Defines the Residual Network (ResNet18); 
   * `mlp.py`: Defines the Multi-Layer Perceptron;
   * `rnn.py`: Defines the Long Short-Term Memory;
 * `outputs`: Folder that holds the saved models and optimization histories, such as `.pth` and `.pkl`;
 * `utils`
   * `attribute.py`: Re-writes getters and setters for nested attributes;
   * `loader.py`: Utility to load datasets and split them into training, validation and testing sets;
   * `objects.py`: Wraps objects instantiation for command line usage;
   * `optimizer.py`: Wraps the optimization task into a single method;  
   * `targets.py`: Implements the objective functions to be optimized.
   
   
---

## Package Guidelines

### Installation

Install all the pre-needed requirements using:

```Python
pip install -r requirements.txt
```

### Data configuration

In order to run the experiments, you can use `torchvision` and `torchtext` to load pre-implemented datasets.

---

## Usage

### Model Training

The first step is to pre-train a Machine Learning architecture. To accomplish such a step, one needs to use the following script:

```Python
python image_model_training.py -h
```

or

```Python
python text_model_training.py -h
```

*Note that line 74 (for image-based) and 75 (for text-based) should be adjusted on `core/model.py` according to the used script.*

### Model Optimization

After conducting the training task, one needs to optimize the weights over the validation set. Please, use the following script to accomplish such a procedure:

```Python
python image_model_optimization.py -h
```

or

```Python
python text_model_optimization.py -h
```

*Note that `-h` invokes the script helper, which assists users in employing the appropriate parameters.*

### Bash Script

Instead of invoking every script to conduct the experiments, it is also possible to use the provided shell script, as follows:

```Bash
./image_pipeline.sh
```

or

```Bash
./text_pipeline.sh
```

Such a script will conduct every step needed to accomplish the experimentation used throughout this paper. Furthermore, one can change any input argument that is defined in the script.

---

## Support

We know that we do our best, but it is inevitable to acknowledge that we make mistakes. If you ever need to report a bug, report a problem, talk to us, please do so! We will be available at our bests at this repository or gustavo.rosa@unesp.br.

---

# PyTorch Basics - Deep Learning Fundamentals

A comprehensive Jupyter notebook covering the fundamental concepts of PyTorch for deep learning beginners.

## 📚 What You'll Learn

This notebook covers essential PyTorch concepts including:

### 🔢 Tensor Fundamentals
- **Scalar Tensors** - Single value tensors
- **Vector Tensors** - 1D arrays of values
- **Matrix Tensors** - 2D arrays with rows and columns
- **Random Tensors** - Creating tensors with random values

### 🛠️ Tensor Operations
- **Zeros and Ones** - Creating tensors filled with specific values
- **Range and Tensor Like** - Using `torch.arange()` and `torch.ones_like()`
- **Identity Matrix** - Creating identity matrices with `torch.eye()`
- **Full Tensors** - Creating tensors filled with specific values
- **Linspace** - Creating evenly spaced values

### 📊 Data Types and Information
- **Data Types** - Working with int, float, and bool tensors
- **Tensor Information** - Shape, dtype, dimensions, and device
- **Shape Management** - Understanding and handling tensor shapes

### 🔄 Tensor Manipulation
- **Reshape** - Changing tensor layout without changing values
- **Stack** - Combining multiple tensors along new dimensions
- **Squeeze and Unsqueeze** - Adding/removing dimensions of size 1
- **Permute** - Rearranging tensor dimensions

## 🚀 Getting Started

### Prerequisites
- Python 3.7+
- PyTorch
- Jupyter Notebook

### Installation
```bash
# Install PyTorch (CPU version)
pip install torch torchvision torchaudio

# Install Jupyter Notebook
pip install jupyter

# Clone this repository
git clone https://github.com/Moiz-Ali-Max/LearnDeepLearning.git

# Navigate to the directory
cd LearnDeepLearning

# Launch Jupyter Notebook
jupyter notebook
```

## 📖 Usage

1. Open `Pytorch_Basics.ipynb` in Jupyter Notebook
2. Run each cell sequentially to understand the concepts
3. Experiment with the code examples
4. Modify parameters to see how they affect the results

## 🎯 Key Concepts Explained

### Why Tensors?
Tensors are the fundamental data structure in PyTorch, similar to NumPy arrays but optimized for deep learning operations with GPU support and automatic differentiation.

### Shape Management
Understanding tensor shapes is crucial for deep learning:
- **Shape Errors**: Most beginner mistakes occur due to mismatched tensor shapes
- **Matrix Multiplication**: Requires compatible shapes (columns of first = rows of second)
- **Model Input**: Different models expect specific input shapes

### Practical Applications
- **Scalar**: Loss values, accuracy metrics
- **Vector**: Feature vectors, embeddings
- **Matrix**: Image data, weight matrices
- **3D+ Tensors**: Batch processing, multi-channel data

## 🤝 Contributing

Feel free to contribute to this project by:
- Adding more examples
- Improving explanations
- Fixing typos
- Adding new PyTorch concepts

## 📝 License

This project is open source and available under the [MIT License](LICENSE).

## 👨‍💻 Author

**Moiz Ali Max**
- GitHub: [@Moiz-Ali-Max](https://github.com/Moiz-Ali-Max)

## 🙏 Acknowledgments

- PyTorch team for the excellent deep learning framework
- The deep learning community for continuous learning resources

---

⭐ **Star this repository if you found it helpful!**

# Using Jax to build Poisson Problem Solver

This repository contains 2 Jupyter Notebook for solving 2D Poisson Equation. `jax_fem_rec.ipynb` is still under construction🚧. 

## Installation

Before running the notebook, ensure you have the required dependencies installed:

```bash
pip install -r requirements.txt
```

## Usage

Here is a brief explanation/code samples on how you might use or test out the solver.

```python
# 
    @jit
    # Define you own custom source term function
    def custom_source_term(x, y):
        return -y * (1 - y) * (1 - x - 0.5 * x**2) * jnp.exp(x + y) - x * (1 - 0.5 * x) * (-3 * y - y**2) * jnp.exp(x + y)

    # Define individual boundary functions
    @jit
    def left_boundary(x, y):
        return -1.5 * y * (1 - y) * jnp.exp(-1 + y)
    @jit
    def right_boundary(x, y):
        return 0.5 * y * (1 - y) * jnp.exp(1 + y)
    @jit
    def bottom_boundary(x, y):
        return -2 * x * (1 - 0.5 * x) * jnp.exp(x - 1)
    @jit
    def top_boundary(x, y):
        return 0.0
    
    #Initialize the size of you own mesh
    N1, N2 = 16, 16

    #Create your own FEM class for Poisson Problem
    myFEM = FEM(N1, N2,
                -1, 1, -1, 1,
                custom_source_term, left_boundary, right_boundary, bottom_boundary, top_boundary)

    # Build up stiffness and mass matrix
    A = myFEM.get_stiffness_mat_jax()
    b = myFEM.get_load_vector_jax()

    # Solve and get your simulated result 
    u = jnp.dot(jnp.linalg.inv(A),b)
    print("u is: ",'\n',u)

    # For more usage please refer to the notebooks for more details
```

## Contributing

We welcome contributions! Please refer to our `Issue` page for guidelines on how to submit improvements or bug fixes.

## License

For details, see the `LICENSE` file in the repository.

## Acknowledgements

We would like to acknowledge Ockimel(https://github.com/Ockimel) for his support and contribution to this project.

# Math Background 
If you are interested in the Math theorem of this repo, please refer to the .pdf file for details.
---




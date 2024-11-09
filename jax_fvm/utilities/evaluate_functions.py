import jax

def CreateFUniformScalar(value):
    def f_uniform(point):
        return value
    f_uniform_batch = jax.vmap(f_uniform)
    return f_uniform_batch
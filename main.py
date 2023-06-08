from tensor import Tensor, GradFn


def main():
    # y = 1 + a * 3 + e ^ (2 / b)
    # y = ((1 + a * 3) + e ^ (2 / b))
    print(GradFn.Add._value_, GradFn.Sub._value_)
    # c = x + y + 4 * z
    x = Tensor(1, requires_grad=True)
    y = Tensor(2, requires_grad=True)
    z = Tensor(4, requires_grad=True)
    e = Tensor(2)
    c = ((x + y) + e * z)
    c.backward()
    print(f"{c=}, {c.grad_fn}, {c.parents}, {c.requires_grad}")
    print(f"dc/dx={x.grad} dc/dy={y.grad} dc/dz={z.grad}")

    # c = 3 * x - 3 * x
    x = Tensor(1, requires_grad=True)
    y = Tensor(3) * x
    z = Tensor(3) * x
    c = y - z
    c.backward()
    print(f"{c=}, dc/dx={x.grad}")

    x = Tensor(2, requires_grad=True)
    y = Tensor(3)
    z = x ** y
    z.backward()
    print(f"{z=} {z.grad_fn} dz/dx={x.grad}")

    x = Tensor(1)
    y = Tensor(2, requires_grad=True)
    z = x / (y ** Tensor(3))
    z.backward()
    print(f"{z=} dz/dy={y.grad}")

    print(Tensor.uniform(0, 1))

if __name__ == "__main__":
    main()

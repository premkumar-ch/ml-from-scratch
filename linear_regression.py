import random

# Original equation: y = 5 + 2*x
# Model: y = a + b*x

X = [4, 35, 43, 25, 1, 31, 28, 50, 50, 40]
Y = [13, 75, 91, 55, 7, 67, 61, 105, 105, 85]

N = len(X)
iterations = 10000000
learning_rate = 0.0000001


def get_rss(a, b):
    rss = 0
    for i in range(N):
        y_pred = a + b * X[i]
        error = y_pred - Y[i]
        rss += error * error
    return rss


def rss_with_a(a, b):
    grad_a = 0
    for i in range(N):
        grad_a += Y[i] - a - b * X[i]
    return -2 * grad_a


def rss_with_b(a, b):
    grad_b = 0
    for i in range(N):
        grad_b += X[i] * (Y[i] - a - b * X[i])
    return -2 * grad_b


a = 0.0
b = 0.0

for i in range(iterations):
    # Recalculate gradients using current a and b
    grad_a = rss_with_a(a, b)
    grad_b = rss_with_b(a, b)

    # Update parameters
    a = a - learning_rate * grad_a
    b = b - learning_rate * grad_b

    if i % 1000000 == 0:
        print(f"Iteration {i}: a={a:.4f}, b={b:.4f}, RSS={get_rss(a, b):.4f}")

print("\nFinal result:")
print(f"a = {a:.4f}")
print(f"b = {b:.4f}")
print(f"RSS = {get_rss(a, b):.4f}")


def gen_data(N):
    X = []
    Y = []

    for i in range(10):
        x = random.randint(1, 50)
        y = 5 + 2 * x

        X.append(x)
        Y.append(y)

    print("X =", X)
    print("Y =", Y)


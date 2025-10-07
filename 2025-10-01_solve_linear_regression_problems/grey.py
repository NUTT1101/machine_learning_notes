import numpy as np

def get_w1(x, r):
    x_bar = np.mean(x)
    r_bar = np.mean(r)
    return (np.mean(x * r) - x_bar * r_bar) / (np.mean(x ** 2) - x_bar ** 2)

def get_w0(x, r):
    return np.mean(r) - get_w1(x, r) * np.mean(x)

def f_of_x(x, w1, w0):
    return w1 * x + w0

if __name__ == "__main__":
    x = np.array([
        17703, 19079, 20620, 22181, 23632, 24911, 26371, 27986, 29467, 30960,
        32226, 33672, 35146, 36572, 37929,39274, 40610, 42063, 43332, 44696,
        46154
    ])

    r = np.array([
        1009, 1102, 1202, 1310, 1399, 1497, 1598, 1707, 1806, 1906,
        2001, 2103, 2209, 2309, 2402,2500, 2601, 2703, 2803, 2902, 3008
    ])

    w1 = get_w1(x, r)
    w0 = get_w0(x, r)

    print(f"w1 = {w1:.2f}")
    print(f"w0 = {w0:.2f}")

    for x in [38162, 21537, 50000]:
        print(f"f({x}) = {f_of_x(x, w1, w0):.2f}")
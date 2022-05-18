import matplotlib.pyplot as plt

def scatter_plotter(spent, payout):
    # Set axes limits
    plt.xlim(0, 1.1 * max(spent))
    plt.ylim(0, 1.1 * max(payout))

    plt.scatter(spent, payout, color="green",
                marker="*", s=30)

    plt.xlabel('Spent - axis')
    plt.ylabel('Payout - axis')

    plt.title('Payout scatter plot')
    plt.legend()
    plt.show()

x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
y = [2, 4, 5, 7, 6, 8, 9, 11, 12, 12]

scatter_plotter(x, y)
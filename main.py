from build_csr_plot import create_combined_csr_plot
from build_soc_plot import create_combined_soc_plot


def main():
    create_combined_csr_plot(plot_name='01-CSR')
    create_combined_soc_plot(plot_name='02-SOC.pdf')


if __name__ == "__main__":
    main()

palletes = {
    "creditas_palette": [
        '#11bb77',
        '#556666',
        '#3377bb'],
    "creditas_green_palette": ['#99ccbb',
                               '#66bb99',
                               '#11bb77',
                               '#228855',
                               '#005533'],
    "creditas_grey_palette": ['#eeeeee',
                              '#cccccc',
                              '#aabbbb',
                              '#778888',
                              '#556666',
                              '#334444',
                              '#111111'],
    "creditas_blue_palette": ['#ddeeff',
                              '#ccddee',
                              '#99bbdd',
                              '#6699dd',
                              '#3377bb',
                              '#005599',
                              '#113377']
}


def palette_roulette(colors, n_start=0):
    n = n_start
    total = len(colors)
    while True:
        yield colors[n]
        n = (n + 1) % total


def get_palette(name="creditas_palette", n_colors=7):
    palette = palette_roulette(palletes[name])

    return [next(palette) for _ in range(n_colors)]

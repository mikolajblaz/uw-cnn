def lighten(image, rng):
    print "Lighten"
    return image


def rotate(image, rng):
    print "Rotate"
    return image


functions = [lighten, rotate]


class Augmentation:
    def __init__(self, rng, fun_list=functions):
        self.fun_list = fun_list
        self.length = len(fun_list)
        self.rng = rng

    def augment_image(self, image):
        rng = self.rng
        rand_fun_idx = rng.randint(0, self.length)
        fun = self.fun_list[rand_fun_idx]
        return fun(image, rng)

    def augment_batch(self, images):
        return images

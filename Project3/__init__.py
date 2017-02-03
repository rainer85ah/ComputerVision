"""
Clase práctica 3.- Detección de objetos.
1 Detección de coches mediante puntos singulares.


class ORB : public Feature2D

Class implementing the ORB (oriented BRIEF) keypoint detector and descriptor extractor, described in [RRKB11].
The algorithm uses FAST in pyramids to detect stable keypoints, selects the strongest features using FAST or Harris response,
finds their orientation using first-order moments and computes the descriptors using BRIEF (where the
coordinates of random point pairs (or k-tuples) are rotated according to the measured orientation).

Al proceso de detección de puntos de interés, extracción de descriptores, extracción de vectores de
votación y construcción del índice de búsqueda por descriptor (usando el cv2.FlannBasedMatcher) le
llamaremos entrenamiento de un detector de objetos basado en votación a la Hough con puntos de
interés.


"""
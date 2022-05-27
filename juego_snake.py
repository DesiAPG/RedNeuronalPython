import pygame
import random
from enum import Enum
from collections import namedtuple
import numpy as np

pygame.init()
font = pygame.font.Font('arial.ttf', 25)


class Direccion(Enum):
    DERECHA = 1
    IZQUIERDA = 2
    ARRIBA = 3
    ABAJO = 4


Point = namedtuple('Point', 'x, y')

# rgb colors
BLANCO = (255, 255, 255)
ROJO = (200, 0, 0)
AZUL1 = (0, 0, 255)
AZUL2 = (0, 100, 255)
NEGRO = (0, 0, 0)

ALTURA_BLOQUE = 20
RAPIDEZ = 15


class serpienteGameAI:

    def __init__(self, w=640, h=480):
        self.w = w
        self.h = h
        # Inicia la pantalla de juego
        self.display = pygame.display.set_mode((self.w, self.h))
        pygame.display.set_caption('Serpiente')
        self.clock = pygame.time.Clock()
        self.reset()

    def reset(self):
        # Inicia el estado del codigo
        self.direccion = Direccion.DERECHA

        self.cabeza = Point(self.w/2, self.h/2)
        self.serpiente = [self.cabeza,
                          Point(self.cabeza.x-ALTURA_BLOQUE, self.cabeza.y),
                          Point(self.cabeza.x-(2*ALTURA_BLOQUE), self.cabeza.y)]

        self.puntaje = 0
        self.comida = None
        self._colocar_comida()
        self.frame_iteration = 0

    def _colocar_comida(self):
        x = random.randint(0, (self.w-ALTURA_BLOQUE) //
                           ALTURA_BLOQUE)*ALTURA_BLOQUE
        y = random.randint(0, (self.h-ALTURA_BLOQUE) //
                           ALTURA_BLOQUE)*ALTURA_BLOQUE
        self.comida = Point(x, y)
        if self.comida in self.serpiente:
            self._colocar_comida()

    def play_step(self, action):
        self.frame_iteration += 1

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()

        # 2. Mover
        self._mover(action)  # Actualiza la cabeza de la serpiente
        self.serpiente.insert(0, self.cabeza)

        # 3. Comprueba si se acabó el juego
        reward = 0
        game_over = False
        if self.hay_colision() or self.frame_iteration > 100*len(self.serpiente):
            game_over = True
            reward = -10
            return reward, game_over, self.puntaje

        # 4. Crea mas comida o seguir moviendo
        if self.cabeza == self.comida:
            self.puntaje += 1
            reward = 10
            self._colocar_comida()
        else:
            self.serpiente.pop()

        # 5. Actualiza la UI
        self._actualizar_ui()
        self.clock.tick(RAPIDEZ)
        # 6. Retorna el game over y el puntaje
        return reward, game_over, self.puntaje

    def hay_colision(self, pt=None):
        if pt is None:
            pt = self.cabeza
        # Si colisiona con la pared
        if pt.x > self.w - ALTURA_BLOQUE or pt.x < 0 or pt.y > self.h - ALTURA_BLOQUE or pt.y < 0:
            return True
        # Si colisiona con ella misma
        if pt in self.serpiente[1:]:
            return True

        return False

    def _actualizar_ui(self):
        self.display.fill(NEGRO)

        for pt in self.serpiente:
            pygame.draw.rect(self.display, AZUL1, pygame.Rect(
                pt.x, pt.y, ALTURA_BLOQUE, ALTURA_BLOQUE))
            pygame.draw.rect(self.display, AZUL2,
                             pygame.Rect(pt.x+4, pt.y+4, 12, 12))

        pygame.draw.rect(self.display, ROJO, pygame.Rect(
            self.comida.x, self.comida.y, ALTURA_BLOQUE, ALTURA_BLOQUE))

        text = font.render("puntaje: " + str(self.puntaje), True, BLANCO)
        self.display.blit(text, [0, 0])
        pygame.display.flip()

    def _mover(self, action):
        # [recto, derecha, izquierda]

        clock_wise = [Direccion.DERECHA, Direccion.ABAJO,
                      Direccion.IZQUIERDA, Direccion.ARRIBA]
        idx = clock_wise.index(self.direccion)

        if np.array_equal(action, [1, 0, 0]):
            new_dir = clock_wise[idx]  # no change
        elif np.array_equal(action, [0, 1, 0]):
            next_idx = (idx + 1) % 4
            # Girar a la derecha r -> d -> l -> u
            new_dir = clock_wise[next_idx]
        else:  # [0, 0, 1]
            next_idx = (idx - 1) % 4
            # Girar a la izquierda r -> u -> l -> d
            new_dir = clock_wise[next_idx]

        self.direccion = new_dir

        x = self.cabeza.x
        y = self.cabeza.y
        if self.direccion == Direccion.DERECHA:
            x += ALTURA_BLOQUE
        elif self.direccion == Direccion.IZQUIERDA:
            x -= ALTURA_BLOQUE
        elif self.direccion == Direccion.ABAJO:
            y += ALTURA_BLOQUE
        elif self.direccion == Direccion.ARRIBA:
            y -= ALTURA_BLOQUE

        self.cabeza = Point(x, y)

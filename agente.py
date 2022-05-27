import torch
import random
import numpy as np
from collections import deque
from juego_snake import serpienteGameAI, Direccion, Point
from modelo import Linear_QNet, QTrainer
from helper import plot

MEMORIA_MAXIMA = 100_000
TAMAñO_BLOQUE = 1000
LR = 0.001


class Agente:

    def __init__(self):
        self.n_juegos = 0
        self.epsilon = 0  # randomness
        self.gamma = 0.9  # discount rate
        self.memory = deque(maxlen=MEMORIA_MAXIMA)  # popIZQUIERDA()
        self.modelo = Linear_QNet(11, 256, 3)
        self.trainer = QTrainer(self.modelo, lr=LR, gamma=self.gamma)

    def get_state(self, juego_serpiente):
        cabeza = juego_serpiente.serpiente[0]
        point_l = Point(cabeza.x - 20, cabeza.y)
        point_r = Point(cabeza.x + 20, cabeza.y)
        point_u = Point(cabeza.x, cabeza.y - 20)
        point_d = Point(cabeza.x, cabeza.y + 20)

        dir_l = juego_serpiente.direccion == Direccion.IZQUIERDA
        dir_r = juego_serpiente.direccion == Direccion.DERECHA
        dir_u = juego_serpiente.direccion == Direccion.ARRIBA
        dir_d = juego_serpiente.direccion == Direccion.ABAJO

        state = [
            # Peligro Recto
            (dir_r and juego_serpiente.hay_colision(point_r)) or
            (dir_l and juego_serpiente.hay_colision(point_l)) or
            (dir_u and juego_serpiente.hay_colision(point_u)) or
            (dir_d and juego_serpiente.hay_colision(point_d)),

            # Peligro derecha
            (dir_u and juego_serpiente.hay_colision(point_r)) or
            (dir_d and juego_serpiente.hay_colision(point_l)) or
            (dir_l and juego_serpiente.hay_colision(point_u)) or
            (dir_r and juego_serpiente.hay_colision(point_d)),

            # Peligro IZQUIERDA
            (dir_d and juego_serpiente.hay_colision(point_r)) or
            (dir_u and juego_serpiente.hay_colision(point_l)) or
            (dir_r and juego_serpiente.hay_colision(point_u)) or
            (dir_l and juego_serpiente.hay_colision(point_d)),

            # Move Direccion
            dir_l,
            dir_r,
            dir_u,
            dir_d,

            # comida location
            juego_serpiente.comida.x < juego_serpiente.cabeza.x,  # comida IZQUIERDA
            juego_serpiente.comida.x > juego_serpiente.cabeza.x,  # comida derecha
            juego_serpiente.comida.y < juego_serpiente.cabeza.y,  # comida arriba
            juego_serpiente.comida.y > juego_serpiente.cabeza.y  # comida abajo
        ]

        return np.array(state, dtype=int)

    def remember(self, state, action, reward, next_state, done):
        # popIZQUIERDA if MEMORIA_MAXIMA is reached
        self.memory.append((state, action, reward, next_state, done))

    def train_long_memory(self):
        if len(self.memory) > TAMAñO_BLOQUE:
            mini_sample = random.sample(
                self.memory, TAMAñO_BLOQUE)  # lista de tuplas
        else:
            mini_sample = self.memory

        states, actions, rewards, next_states, done = zip(*mini_sample)
        self.trainer.train_step(
            states, actions, rewards, next_states, done)

    def train_short_memory(self, state, action, reward, next_state, done):
        self.trainer.train_step(state, action, reward, next_state, done)

    def get_action(self, state):
        # movimientos aleatorios
        self.epsilon = 80 - self.n_juegos
        final_move = [0, 0, 0]
        if random.randint(0, 200) < self.epsilon:
            move = random.randint(0, 2)
            final_move[move] = 1
        else:
            state0 = torch.tensor(state, dtype=torch.float)
            prediction = self.modelo(state0)
            move = torch.argmax(prediction).item()
            final_move[move] = 1

        return final_move


def train():
    graficar_puntaje = []
    plot_mean_scores = []
    puntaje_total = 0
    record = 0
    agente = Agente()
    juego_serpiente = serpienteGameAI()
    while True:
        # Obtiene el anterior estado
        estado_antiguo = agente.get_state(juego_serpiente)

        # Obtiene el movimiento
        final_move = agente.get_action(estado_antiguo)

        # Hace el movimiento y actualiza el estado
        reward, done, puntaje = juego_serpiente.play_step(final_move)
        state_new = agente.get_state(juego_serpiente)

        # Entrena el short memory
        agente.train_short_memory(
            estado_antiguo, final_move, reward, state_new, done)

        # Recuerda
        agente.remember(estado_antiguo, final_move, reward, state_new, done)

        if done:
            # Entrena el long memory, Y registra en la grafica
            juego_serpiente.reset()
            agente.n_juegos += 1
            agente.train_long_memory()

            if puntaje > record:
                record = puntaje
                agente.modelo.guardar()

            print('juego_serpiente', agente.n_juegos,
                  'Puntaje', puntaje, 'Record:', record)

            graficar_puntaje.append(puntaje)
            puntaje_total += puntaje
            mean_score = puntaje_total / agente.n_juegos
            plot_mean_scores.append(mean_score)
            plot(graficar_puntaje, plot_mean_scores)


if __name__ == '__main__':
    train()

import pygame
import sys
import time
import jugador  # Asegúrate que este sea el nombre del archivo con la lógica del juego

pygame.init()
tamano = ancho, alto = 1024, 720

# Colores
negro = (0, 0, 0)
blanco = (255, 255, 255)

pantalla = pygame.display.set_mode(tamano)

fuenteMedia = pygame.font.Font("OpenSans-Regular.ttf", 28)
fuenteGrande = pygame.font.Font("OpenSans-Regular.ttf", 40)
fuenteMovimiento = pygame.font.Font("OpenSans-Regular.ttf", 60)

usuario = None
tablero = jugador.estado_inicial()
turno_ia = False

while True:

    for evento in pygame.event.get():
        if evento.type == pygame.QUIT:
            sys.exit()

    pantalla.fill(negro)

    # Dejar que el usuario elija un jugador
    if usuario is None:

        # Título
        titulo = fuenteGrande.render("Jugar Triqui", True, blanco)
        rectTitulo = titulo.get_rect()
        rectTitulo.center = (ancho / 2, 50)
        pantalla.blit(titulo, rectTitulo)

        # Botones
        botonX = pygame.Rect(ancho / 10, alto / 2, ancho / 4, 50)
        textoX = fuenteMedia.render("Jugar como X", True, negro)
        rectTextoX = textoX.get_rect()
        rectTextoX.center = botonX.center
        pygame.draw.rect(pantalla, blanco, botonX)
        pantalla.blit(textoX, rectTextoX)

        botonO = pygame.Rect(5 * (ancho / 10), alto / 2, ancho / 4, 50)
        textoO = fuenteMedia.render("Jugar como O", True, negro)
        rectTextoO = textoO.get_rect()
        rectTextoO.center = botonO.center
        pygame.draw.rect(pantalla, blanco, botonO)
        pantalla.blit(textoO, rectTextoO)

        # Detectar clics
        clic, _, _ = pygame.mouse.get_pressed()
        if clic == 1:
            raton = pygame.mouse.get_pos()
            if botonX.collidepoint(raton):
                time.sleep(0.2)
                usuario = jugador.X
            elif botonO.collidepoint(raton):
                time.sleep(0.2)
                usuario = jugador.O

    else:
        # Dibujar tablero
        tamano_casilla = 80
        origen = (ancho / 2 - 1.5 * tamano_casilla, alto / 2 - 1.5 * tamano_casilla)
        casillas = []
        for i in range(3):
            fila = []
            for j in range(3):
                rect = pygame.Rect(
                    origen[0] + j * tamano_casilla,
                    origen[1] + i * tamano_casilla,
                    tamano_casilla, tamano_casilla
                )
                pygame.draw.rect(pantalla, blanco, rect, 3)

                if tablero[i][j] != jugador.VACIO:
                    movimiento = fuenteMovimiento.render(tablero[i][j], True, blanco)
                    rectMovimiento = movimiento.get_rect()
                    rectMovimiento.center = rect.center
                    pantalla.blit(movimiento, rectMovimiento)
                fila.append(rect)
            casillas.append(fila)

        juego_terminado = jugador.final(tablero)
        turno_actual = jugador.jugador(tablero)

        # Mostrar título
        if juego_terminado:
            ganador = jugador.ganador(tablero)
            if ganador is None:
                texto = "Fin del juego: Empate."
            else:
                texto = f"Fin del juego: {ganador} gana."
        elif usuario == turno_actual:
            texto = f"Turno: {usuario}"
        else:
            texto = "Pensando..."

        titulo = fuenteGrande.render(texto, True, blanco)
        rectTitulo = titulo.get_rect()
        rectTitulo.center = (ancho / 2, 30)
        pantalla.blit(titulo, rectTitulo)

        # Movimiento IA
        if usuario != turno_actual and not juego_terminado:
            if turno_ia:
                time.sleep(0.5)
                movimiento = jugador.minimax(tablero)
                tablero = jugador.resultado(tablero, movimiento)
                turno_ia = False
            else:
                turno_ia = True

        # Movimiento del usuario
        clic, _, _ = pygame.mouse.get_pressed()
        if clic == 1 and usuario == turno_actual and not juego_terminado:
            raton = pygame.mouse.get_pos()
            for i in range(3):
                for j in range(3):
                    if tablero[i][j] == jugador.VACIO and casillas[i][j].collidepoint(raton):
                        tablero = jugador.resultado(tablero, (i, j))

        # Botón de jugar otra vez
        if juego_terminado:
            botonReiniciar = pygame.Rect(ancho / 3, alto - 65, ancho / 3, 50)
            textoReiniciar = fuenteMedia.render("Jugar de nuevo", True, negro)
            rectReiniciar = textoReiniciar.get_rect()
            rectReiniciar.center = botonReiniciar.center
            pygame.draw.rect(pantalla, blanco, botonReiniciar)
            pantalla.blit(textoReiniciar, rectReiniciar)
            clic, _, _ = pygame.mouse.get_pressed()
            if clic == 1:
                raton = pygame.mouse.get_pos()
                if botonReiniciar.collidepoint(raton):
                    time.sleep(0.2)
                    usuario = None
                    tablero = jugador.estado_inicial()
                    turno_ia = False

    pygame.display.flip()

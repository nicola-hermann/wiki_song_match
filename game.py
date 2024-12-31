import pygame
import wikipedia
import threading
import requests
from pygame.locals import *
from PIL import Image, ImageFilter
import numpy as np

background_image = Image.open(r"assets\background.png")
blurred_background = background_image.filter(ImageFilter.GaussianBlur(10))
blurred_background.save("assets/blurred_background.png")

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH, HEIGHT = 800, 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Wikipedia Game")

# Fonts and colors
FONT = pygame.font.Font(None, 36)
SMALL_FONT = pygame.font.Font(None, 28)
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
GRAY = (200, 200, 200)
BLUE = (0, 120, 215)

# Load the background image
BACKGROUND_IMAGE = pygame.image.load(r"assets\blurred_background.png")
BACKGROUND_IMAGE = pygame.transform.scale(BACKGROUND_IMAGE, (WIDTH, HEIGHT))

BASE_URL = "http://127.0.0.1:5000"


overlay = pygame.Surface((WIDTH, HEIGHT))
overlay.set_alpha(100)  # Adjust transparency (0-255)
overlay.fill((0, 0, 0))  # Black overlay


# Text input class
class TextBox:
    def __init__(self, x, y, w, h, placeholder):
        self.rect = pygame.Rect(x, y, w, h)
        self.color = GRAY
        self.text = ""
        self.placeholder = placeholder
        self.active = False

    def handle_event(self, event):
        if event.type == MOUSEBUTTONDOWN:
            self.active = self.rect.collidepoint(event.pos)
        if event.type == KEYDOWN and self.active:
            if event.key == K_BACKSPACE:
                self.text = self.text[:-1]
            else:
                self.text += event.unicode

    def draw(self, screen):
        border_color = BLUE if self.active else BLACK
        pygame.draw.rect(screen, self.color, self.rect, border_radius=10)
        pygame.draw.rect(screen, border_color, self.rect, 2, border_radius=10)
        text_surface = FONT.render(self.text or self.placeholder, True, BLACK)
        screen.blit(text_surface, (self.rect.x + 10, self.rect.y + 10))


def render_multiline_text(text, x, y, width, font, color, surface):
    """
    Renders text across multiple lines if it exceeds the given width.
    """
    words = text.split(" ")
    lines = []
    current_line = ""

    for word in words:
        test_line = f"{current_line} {word}".strip()
        if font.size(test_line)[0] <= width:
            current_line = test_line
        else:
            lines.append(current_line)
            current_line = word
    lines.append(current_line)  # Append the last line

    for i, line in enumerate(lines):
        line_surface = font.render(line, True, color)
        surface.blit(line_surface, (x, y + i * font.get_linesize()))


def display_result_screen(screen, result, player_artist, player_title):
    bot_artist, bot_title, distance1, distance2 = result

    # Set up the result screen
    screen.fill(BLACK)
    result_text = FONT.render("Results", True, WHITE)
    screen.blit(result_text, (WIDTH // 2 - result_text.get_width() // 2, 50))

    player_choice_text = FONT.render(
        f"Player Choice: {player_artist} - {player_title}", True, WHITE
    )
    screen.blit(player_choice_text, (20, 120))

    bot_choice_text = FONT.render(
        f"Bot Choice: {bot_artist} - {bot_title}", True, WHITE
    )
    screen.blit(bot_choice_text, (20, 180))

    distance_text1 = FONT.render(f"Distance (Player): {distance2:.5f}", True, WHITE)
    screen.blit(distance_text1, (20, 240))

    distance_text2 = FONT.render(f"Distance (Bot): {distance1:.5f}", True, WHITE)
    screen.blit(distance_text2, (20, 300))

    winner_text = FONT.render("Winner: ", True, WHITE)
    if distance1 < distance2:
        winner_text = FONT.render("Winner: Bot", True, WHITE)
    else:
        winner_text = FONT.render("Winner: Player", True, WHITE)
    screen.blit(winner_text, (WIDTH // 2 - winner_text.get_width() // 2, 400))

    # Draw the "Stop" button
    pygame.draw.rect(screen, stop_button_color, stop_button_rect, border_radius=10)
    stop_button_text = FONT.render("Stop", True, BLACK)
    text_rect = stop_button_text.get_rect(center=stop_button_rect.center)
    screen.blit(stop_button_text, text_rect)

    # Draw the "Replay" button
    pygame.draw.rect(screen, replay_button_color, replay_button_rect, border_radius=10)
    replay_button_text = FONT.render("Replay", True, BLACK)
    replay_text_rect = replay_button_text.get_rect(center=replay_button_rect.center)
    screen.blit(replay_button_text, replay_text_rect)

    pygame.display.flip()

    # Handle events for both "Stop" and "Replay" buttons
    stop_button_clicked = False
    replay_button_clicked = False
    while not stop_button_clicked and not replay_button_clicked:
        for event in pygame.event.get():
            if event.type == QUIT:
                pygame.quit()
                return
            if event.type == MOUSEBUTTONDOWN:
                if stop_button_rect.collidepoint(event.pos):
                    stop_button_clicked = True  # Stop the game
                    pygame.quit()
                elif replay_button_rect.collidepoint(event.pos):
                    replay_button_clicked = True  # Replay the game

            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    stop_button_clicked = True
                    pygame.quit()

    # If "Replay" is clicked, restart the game
    if replay_button_clicked:
        return True  # Indicate to restart the game
    return False


# API call placeholders
def match_request(url, result):
    response = requests.get(url)
    result["response"] = response.json()


def lyrics_request(url, data, result):
    response = requests.post(url, json=data)
    result["response"] = response.json()


# Result calculation placeholder
def calculate_result(api1_result, api2_result):
    res1 = api1_result["response"]["data"][0]
    res2 = api2_result["response"]

    lyrics_embedding = res1["lyrics_embedding"]
    wiki_embedding = res1["wiki_embedding"]
    artist = res1["artist"]
    title = res1["title"]

    player_embedding = res2.get("embedding")
    if player_embedding is None:
        raise NameError("Song could not be found on lyrics.com")

    # Calculate the euclidean distance between the two embeddings
    distance1 = np.linalg.norm(np.array(lyrics_embedding) - np.array(wiki_embedding))
    distance2 = np.linalg.norm(np.array(player_embedding) - np.array(wiki_embedding))

    return artist, title, distance1, distance2


def submit(song, artist, api2_result):
    thread2 = threading.Thread(
        target=lyrics_request,
        args=(
            f"{BASE_URL}/lyrics_keywords",
            {"song": song, "artist": artist},
            api2_result,
        ),
    )
    thread2.start()
    return thread2


# Button
button_rect = pygame.Rect(350, 520, 100, 40)
button_color = GRAY

stop_button_rect = pygame.Rect(WIDTH // 2 - 50, 480, 100, 40)
stop_button_color = GRAY

replay_button_rect = pygame.Rect(WIDTH // 2 - 50, 540, 100, 40)
replay_button_color = GRAY


def start_game():
    # Textboxes
    title_box = TextBox(200, 400, 400, 40, "Enter Title")
    artist_box = TextBox(200, 460, 400, 40, "Enter Artist")
    textboxes = [title_box, artist_box]
    current_textbox = 0
    # Track the state of both API calls
    thread1_finished = False
    thread2_started = False
    thread2_finished = False

    running = True

    # Get Wikipedia article
    article_summary = None
    while article_summary is None:
        try:
            article_title = wikipedia.random()
            article = wikipedia.page(article_title)
            article_summary = article.summary
        except:
            continue

    # API results and threading
    api1_result = {}
    api2_result = {}
    thread1 = threading.Thread(
        target=match_request,
        args=(f"{BASE_URL}/match/{article_title}", api1_result),
    )
    thread1.start()

    while running:
        screen.blit(BACKGROUND_IMAGE, (0, 0))
        screen.blit(overlay, (0, 0))

        # Display article information
        title_surface = FONT.render(f"Wikipedia Article: {article_title}", True, WHITE)
        screen.blit(title_surface, (20, 20))

        summary_y_start = 60
        render_multiline_text(
            article_summary,
            20,
            summary_y_start,
            WIDTH - 40,
            SMALL_FONT,
            WHITE,
            screen,
        )

        for i, box in enumerate(textboxes):
            box.active = i == current_textbox
            box.draw(screen)

        # Draw the rounded button
        pygame.draw.rect(screen, button_color, button_rect, border_radius=10)

        # Center the "Submit" text inside the button
        button_text = FONT.render("Submit", True, BLACK)
        text_rect = button_text.get_rect(center=button_rect.center)
        screen.blit(button_text, text_rect)

        for event in pygame.event.get():
            if event.type == QUIT:
                running = False

            if event.type == KEYDOWN:
                if event.key == K_TAB:
                    current_textbox = (current_textbox + 1) % len(textboxes)
                elif event.key == K_RETURN:  # Press Enter to submit
                    if title_box.text and artist_box.text:
                        if not thread2_started:
                            thread2 = submit(
                                title_box.text, artist_box.text, api2_result
                            )
                            thread2_started = True

            textboxes[current_textbox].handle_event(event)

            if event.type == MOUSEBUTTONDOWN:
                if button_rect.collidepoint(event.pos):
                    if title_box.text and artist_box.text:
                        if not thread2_started:
                            thread2 = submit(
                                title_box.text, artist_box.text, api2_result
                            )
                            thread2_started = True

        # Check the state of thread1
        if not thread1.is_alive() and not thread1_finished:
            thread1_finished = True
            try:
                api_res = api1_result["response"]
            except Exception as e:
                print("API not availlable")

        # Check the state of thread2
        if thread2_started and not thread2.is_alive() and not thread2_finished:
            thread2_finished = True
            try:
                api_res = api2_result["response"]
            except Exception as e:
                print("API not availlable")

        # If both threads are done, calculate the result
        if thread1_finished and thread2_finished:
            result = calculate_result(api1_result, api2_result)
            if display_result_screen(screen, result, artist_box.text, title_box.text):
                return True

        pygame.display.flip()

    pygame.quit()


while start_game():
    pass

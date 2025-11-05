import atexit
import base64
import json
import queue
import threading

import cv2
import numpy as np
from flask import Flask, Response, render_template


class EnvMonitorBase:
    def __init__(self, config):
        self._config = config
        self._obs_queue = queue.Queue()
        self._info_queue = queue.Queue()
        self._thread = threading.Thread(target=self._run_server)
        self._thread.start()
        atexit.register(self.stop)

    def _run_server(self):
        app = Flask(__name__, template_folder="templates")

        @app.route("/")
        def index():
            return render_template("index.html")

        @app.route("/stream")
        def stream():
            def generate():
                while True:
                    obs = self._obs_queue.get()
                    info = self._info_queue.get()
                    frame = self._render(obs, info)
                    yield f"data: {json.dumps(frame)}\n\n"
                    self._obs_queue.task_done()
                    self._info_queue.task_done()

            return Response(generate(), mimetype="text/event-stream")

        app.run(
            host="0.0.0.0",
            port=self._config.world.carla_port + 7000,
            use_reloader=False,
        )

    def _render_info(self, info):
        rendered_info = {}
        for key, value in info.items():
            if isinstance(value, (float, int, bool)):
                rendered_info[key] = value
            elif isinstance(value, np.number):
                rendered_info[key] = value.item()
            elif isinstance(value, np.ndarray) and value.ndim == 1:
                rendered_info[key] = value.tolist()
            else:
                rendered_info[key] = str(value)
        return rendered_info

    def _render_images(self, obs):
        images = []
        display_config = self._config.display
        if display_config.enable and display_config.render_keys:
            for key in display_config.render_keys:
                if key in obs:
                    img = obs[key]
                    if len(img.shape) == 2:
                        img = np.repeat(img[:, :, np.newaxis], 3, axis=2)
                    else:
                        img = img[:, :, ::-1]
                    _, img_encoded = cv2.imencode(".webp", img)
                    img_base64 = base64.b64encode(img_encoded).decode("utf-8")
                    images.append({"key": key, "image": img_base64})
        return images

    def _render(self, obs, info):
        return {"images": self._render_images(obs), "info": self._render_info(info)}

    def stop(self):
        if self._thread and self._thread.is_alive():
            self._thread.join()

    def __del__(self):
        self.stop()


class EnvMonitorOpenCV(EnvMonitorBase):
    def render(self, obs, info):
        if not self._obs_queue.full():
            self._obs_queue.put(obs)
        if not self._info_queue.full():
            self._info_queue.put(info)


class EnvMonitorNull:
    def __init__(self, *args, **kwargs):
        pass

    def render(self, *args, **kwargs):
        pass

    def stop(self):
        pass


class EnvMonitorPygame:
    def __init__(self, config):
        self._config = config
        display_config = getattr(self._config, "display", None)
        if display_config is None:
            raise ValueError("Missing display configuration for pygame monitor.")

        try:
            import pygame
        except ImportError as exc:  # pragma: no cover - executed only when pygame is missing
            raise RuntimeError("Pygame backend requested but pygame is not installed.") from exc

        self._pygame = pygame
        pygame.init()
        pygame.font.init()

        self._render_keys = list(getattr(display_config, "render_keys", ()))
        self._image_size = int(getattr(display_config, "image_size", 512))
        self._window_title = getattr(display_config, "window_title", "CARLA Env Monitor")
        self._max_fps = float(getattr(display_config, "max_fps", 30.0))
        self._info_panel_height = int(getattr(display_config, "info_panel_height", 200))
        self._info_max_lines = int(getattr(display_config, "info_max_lines", 20))
        self._font_name = getattr(display_config, "font_name", "Consolas")
        self._font_size = int(getattr(display_config, "font_size", 16))

        width = max(len(self._render_keys), 1) * self._image_size
        height = self._image_size + max(self._info_panel_height, 0)

        self._screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption(self._window_title)
        self._font = pygame.font.SysFont(self._font_name, self._font_size)
        self._clock = pygame.time.Clock()
        self._background_color = getattr(display_config, "background_color", (0, 0, 0))
        self._info_color = getattr(display_config, "info_text_color", (255, 255, 255))

        self._closed = False
        atexit.register(self.stop)

    def render(self, obs, info):
        if self._closed:
            return
        pygame = self._pygame

        # Handle window events to keep the window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                self.stop()
                return

        self._screen.fill(self._background_color)
        for idx, key in enumerate(self._render_keys):
            if key not in obs:
                continue
            image = self._prepare_image(obs[key])
            if image is None:
                continue
            surface = pygame.surfarray.make_surface(np.transpose(image, (1, 0, 2)))
            if surface is None:
                continue
            surface = pygame.transform.smoothscale(surface, (self._image_size, self._image_size))
            x_offset = idx * self._image_size
            self._screen.blit(surface, (x_offset, 0))
            # Draw label below the image
            label_surface = self._font.render(str(key), True, self._info_color)
            label_rect = label_surface.get_rect()
            label_rect.topleft = (x_offset + 5, self._image_size + 5)
            self._screen.blit(label_surface, label_rect)

        info_start_y = self._image_size + self._font_size + 10
        rendered_info = self._render_info(info)
        for line_idx, (k, v) in enumerate(rendered_info.items()):
            if line_idx >= self._info_max_lines:
                break
            text = f"{k}: {v}"
            text_surface = self._font.render(text, True, self._info_color)
            text_rect = text_surface.get_rect()
            text_rect.topleft = (5, info_start_y + line_idx * (self._font_size + 4))
            self._screen.blit(text_surface, text_rect)

        pygame.display.flip()
        if self._max_fps > 0:
            self._clock.tick(self._max_fps)

    def stop(self):
        if self._closed:
            return
        self._closed = True
        pygame = self._pygame
        pygame.display.quit()
        pygame.quit()

    def _prepare_image(self, image):
        if image is None:
            return None
        arr = np.asarray(image)
        if arr.ndim == 2:
            arr = np.repeat(arr[:, :, np.newaxis], 3, axis=2)
        elif arr.ndim == 3 and arr.shape[2] == 4:
            arr = arr[:, :, :3]
        if arr.dtype != np.uint8:
            arr = np.clip(arr, 0, 255)
            max_val = float(np.max(arr)) if arr.size > 0 else 1.0
            if max_val <= 1.0:
                arr = arr * 255.0
            arr = arr.astype(np.uint8)
        return arr

    def _render_info(self, info):
        rendered_info = {}
        for key, value in info.items():
            if isinstance(value, (float, int, bool)):
                rendered_info[key] = value
            elif isinstance(value, np.number):
                rendered_info[key] = value.item()
            elif isinstance(value, np.ndarray) and value.ndim == 1:
                if value.dtype == np.bool_:
                    rendered_info[key] = value.astype(bool).tolist()
                elif np.issubdtype(value.dtype, np.integer):
                    rendered_info[key] = value.astype(int).tolist()
                elif np.issubdtype(value.dtype, np.floating):
                    rendered_info[key] = np.round(value.astype(np.float64), 3).tolist()
                else:
                    rendered_info[key] = value.tolist()
            else:
                rendered_info[key] = str(value)
        return rendered_info


def create_env_monitor(config):
    display_config = getattr(config, "display", None)
    if display_config is None or not getattr(display_config, "enable", False):
        return EnvMonitorNull()

    backend = getattr(display_config, "backend", "web")
    backend = backend.lower() if isinstance(backend, str) else "web"

    if backend in {"pygame", "window"}:
        return EnvMonitorPygame(config)
    if backend in {"opencv", "web", "flask"}:
        return EnvMonitorOpenCV(config)
    raise ValueError(f"Unsupported display backend '{backend}'.")

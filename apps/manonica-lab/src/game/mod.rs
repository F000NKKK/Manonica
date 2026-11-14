//! Playable voxel demo. OS input stays on winit; game state and rendering share an owned worker.
mod effects;
mod hud;
mod mesh;
mod play;
mod world;
use manonica_world::{
    WorldResult,
    runtime::{EntityId, GameRuntime},
};
use meridian_sdk::{
    AppHandler, FlyControls, InputState, KeyCode, MouseButton, Window, fly_controls,
};
use std::{
    collections::VecDeque,
    sync::{
        Arc, Mutex,
        atomic::{AtomicBool, Ordering},
    },
    thread::{self, JoinHandle},
    time::Duration,
};
#[derive(Clone, Copy)]
enum Action {
    Diagnostics,
    Jump,
    Break,
    Use,
    Select(usize),
    Save,
    Load,
    Respawn,
}
struct Input {
    controls: FlyControls,
    actions: VecDeque<Action>,
    size: [u32; 2],
    locked: bool,
}
impl Input {
    fn push(&mut self, action: Action) {
        if self.actions.len() < 64 {
            self.actions.push_back(action);
        }
    }
}
fn lock<T>(value: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    value.lock().unwrap_or_else(|e| e.into_inner())
}
pub struct GameApp {
    initial: Option<(GameRuntime, EntityId)>,
    input: Arc<Mutex<Input>>,
    running: Arc<AtomicBool>,
    failed: Arc<AtomicBool>,
    worker: Option<JoinHandle<()>>,
    smoke_frames: Option<u64>,
    showcase: bool,
}
impl GameApp {
    pub fn new(smoke_frames: Option<u64>, showcase: bool) -> WorldResult<Self> {
        Ok(Self {
            initial: Some(world::generate()?),
            input: Arc::new(Mutex::new(Input {
                controls: FlyControls::default(),
                actions: VecDeque::new(),
                size: [1280, 800],
                locked: true,
            })),
            running: Arc::new(AtomicBool::new(true)),
            failed: Arc::new(AtomicBool::new(false)),
            worker: None,
            smoke_frames,
            showcase,
        })
    }
}
impl AppHandler for GameApp {
    fn on_ready(&mut self, window: &Window) {
        window.set_cursor_grabbed(true);
        let initial = self.initial.take().expect("game starts once");
        let input = self.input.clone();
        let running = self.running.clone();
        let failed = self.failed.clone();
        let window = window.clone();
        let frames = self.smoke_frames;
        let showcase = self.showcase;
        self.worker = Some(thread::spawn(move || {
            if let Err(e) = play::run(window, initial, input, running.clone(), frames, showcase) {
                eprintln!("Game stopped: {e}");
                failed.store(true, Ordering::Release);
            }
            running.store(false, Ordering::Release);
        }));
    }
    fn on_redraw(&mut self, window: &Window, input: &InputState) {
        if self.smoke_frames.is_some() && self.worker.as_ref().is_some_and(JoinHandle::is_finished)
        {
            let result = self.worker.take().unwrap().join();
            std::process::exit(i32::from(
                result.is_err() || self.failed.load(Ordering::Acquire),
            ));
        }
        {
            let mut state = lock(&self.input);
            if !window.surface_target().has_focus() {
                state.locked = false;
                window.set_cursor_grabbed(false);
                state.controls = FlyControls::default();
                state.actions.clear();
            } else {
                if input.was_key_pressed(KeyCode::Escape) {
                    state.locked = !state.locked;
                    window.set_cursor_grabbed(state.locked);
                }
                let mut controls = if state.locked {
                    fly_controls(input)
                } else {
                    FlyControls::default()
                };
                controls.look.0 += state.controls.look.0;
                controls.look.1 += state.controls.look.1;
                state.controls = controls;
                if state.locked {
                    if input.was_key_pressed(KeyCode::Space) {
                        state.push(Action::Jump);
                    }
                    if input.was_mouse_button_pressed(MouseButton::Left) {
                        state.push(Action::Break);
                    }
                    if input.was_mouse_button_pressed(MouseButton::Right) {
                        state.push(Action::Use);
                    }
                    for (i, key) in [
                        KeyCode::Digit1,
                        KeyCode::Digit2,
                        KeyCode::Digit3,
                        KeyCode::Digit4,
                        KeyCode::Digit5,
                        KeyCode::Digit6,
                    ]
                    .into_iter()
                    .enumerate()
                    {
                        if input.was_key_pressed(key) {
                            state.push(Action::Select(i));
                        }
                    }
                    if input.was_key_pressed(KeyCode::F3) {
                        state.push(Action::Diagnostics);
                    }
                    if input.was_key_pressed(KeyCode::R) {
                        state.push(Action::Respawn);
                    }
                }
                if input.was_key_pressed(KeyCode::K) {
                    state.push(Action::Save);
                }
                if input.was_key_pressed(KeyCode::L) {
                    state.push(Action::Load);
                }
            }
        }
        thread::sleep(Duration::from_millis(2));
    }
    fn on_resized(&mut self, width: u32, height: u32) {
        lock(&self.input).size = [width, height];
    }
    fn on_close_requested(&mut self) -> bool {
        self.running.store(false, Ordering::Release);
        true
    }
    fn ready_to_close(&self) -> bool {
        self.worker.as_ref().is_none_or(JoinHandle::is_finished)
    }
}
impl Drop for GameApp {
    fn drop(&mut self) {
        self.running.store(false, Ordering::Release);
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
    }
}

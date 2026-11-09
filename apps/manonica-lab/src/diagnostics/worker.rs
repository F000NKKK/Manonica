//! Lossy, bounded CPU rasterization; no drawing work runs on the graphics thread.
use super::plot::Plot;
use meridian_sdk::ImageAsset;
use std::{
    sync::mpsc::{self, Receiver, SyncSender},
    thread::{self, JoinHandle},
};
pub(super) struct HudWorker {
    sender: Option<SyncSender<Plot>>,
    receiver: Receiver<ImageAsset>,
    thread: Option<JoinHandle<()>>,
}
impl HudWorker {
    pub fn new() -> Self {
        let (sender, requests) = mpsc::sync_channel::<Plot>(1);
        let (results, receiver) = mpsc::sync_channel(1);
        let thread = thread::Builder::new()
            .name("manonica-debug-hud".into())
            .spawn(move || {
                while let Ok(plot) = requests.recv() {
                    if let Some(metrics) = plot.latest {
                        let _ = results.try_send(plot.render_image(metrics));
                    }
                }
            })
            .expect("failed to start HUD rasterizer");
        Self {
            sender: Some(sender),
            receiver,
            thread: Some(thread),
        }
    }
    pub fn offer(&self, plot: &Plot) -> bool {
        self.sender
            .as_ref()
            .is_some_and(|sender| sender.try_send(plot.clone()).is_ok())
    }
    pub fn latest(&self) -> Option<ImageAsset> {
        let mut image = None;
        while let Ok(next) = self.receiver.try_recv() {
            image = Some(next);
        }
        image
    }
}
impl Drop for HudWorker {
    fn drop(&mut self) {
        self.sender.take();
        if let Some(thread) = self.thread.take() {
            let _ = thread.join();
        }
    }
}

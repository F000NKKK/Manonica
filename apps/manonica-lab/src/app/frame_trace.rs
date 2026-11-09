//! Opt-in, buffered raw frame timings for separating actual sources of jitter.
use std::{
    fs::File,
    io::{self, BufWriter, Write},
    path::PathBuf,
};
pub(super) struct FrameSample {
    pub interval_ms: f64,
    pub render_ms: f64,
    pub input_ms: f64,
    pub update_ms: f64,
    pub overlays_ms: f64,
    pub draw_present_ms: f64,
    pub gpu_batch_ms: f64,
    pub batch_steps: u32,
    pub simulation_steps: u64,
}
pub(super) struct FrameTrace {
    writer: BufWriter<File>,
    frame: u64,
}
impl FrameTrace {
    pub fn from_environment() -> io::Result<Option<Self>> {
        let Some(path) = std::env::var_os("MANONICA_FRAME_TRACE") else {
            return Ok(None);
        };
        let file = File::create_new(PathBuf::from(path))?;
        let mut writer = BufWriter::new(file);
        writeln!(
            writer,
            "frame,interval_ms,render_ms,input_ms,update_ms,overlays_ms,draw_present_ms,gpu_batch_ms,batch_steps,simulation_steps"
        )?;
        Ok(Some(Self { writer, frame: 0 }))
    }
    pub fn record(&mut self, sample: FrameSample) -> io::Result<()> {
        self.frame += 1;
        writeln!(
            self.writer,
            "{},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{},{}",
            self.frame,
            sample.interval_ms,
            sample.render_ms,
            sample.input_ms,
            sample.update_ms,
            sample.overlays_ms,
            sample.draw_present_ms,
            sample.gpu_batch_ms,
            sample.batch_steps,
            sample.simulation_steps
        )
    }
    pub fn flush(&mut self) -> io::Result<()> {
        self.writer.flush()
    }
}

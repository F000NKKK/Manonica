#![forbid(unsafe_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut frames = None;
    let mut showcase = false;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--smoke-frames" => frames = Some(args.next().ok_or("missing frame count")?.parse()?),
            "--showcase" => showcase = true,
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    let app = manonica_lab::game::GameApp::new(frames, showcase)?;
    println!(
        "Manonica 3D: WASD, mouse, Shift sprint, Space jump; LMB break, RMB use; 1-6 hotbar; Esc cursor; K save, L load."
    );
    meridian_sdk::run_windowed_app("Manonica — Arcane Frontier", 1280, 800, app)?;
    Ok(())
}

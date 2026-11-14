#![forbid(unsafe_code)]
fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut frames = None;
    let mut capture = None;
    let mut at = 0.8;
    let mut scroll = 4_usize;
    let mut showcase = false;
    let mut profile = false;
    let mut physics_hz = 40;
    let mut args = std::env::args().skip(1);
    while let Some(arg) = args.next() {
        match arg.as_str() {
            "--smoke-frames" => frames = Some(args.next().ok_or("missing frame count")?.parse()?),
            "--showcase" => showcase = true,
            "--profile-physics" => profile = true,
            "--physics-hz" => {
                physics_hz = args.next().ok_or("missing physics frequency")?.parse()?
            }
            "--capture" => {
                capture = Some(std::path::PathBuf::from(
                    args.next().ok_or("missing capture path")?,
                ))
            }
            "--at" => at = args.next().ok_or("missing capture time")?.parse()?,
            "--scroll" => scroll = args.next().ok_or("missing scroll number")?.parse()?,
            _ => return Err(format!("unknown argument {arg}").into()),
        }
    }
    if profile {
        return manonica_lab::game::profile(
            at,
            scroll.checked_sub(1).ok_or("scroll must be 4..6")?,
            physics_hz,
        );
    }
    if let Some(path) = capture {
        return manonica_lab::game::capture(
            &path,
            at,
            scroll.checked_sub(1).ok_or("scroll must be 4..6")?,
        );
    }
    let app = manonica_lab::game::GameApp::new(frames, showcase)?;
    println!(
        "Manonica 3D: WASD, mouse, Shift sprint, Space jump; LMB break, RMB use; 1-6 hotbar; Esc cursor; K save, L load."
    );
    meridian_sdk::run_windowed_app("Manonica — Arcane Frontier", 1280, 800, app)?;
    Ok(())
}

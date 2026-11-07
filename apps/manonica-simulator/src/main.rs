use manonica_physics::{Crystal, Reservoir, diffuse_1d, step_crystal};

fn main() {
    let mut state = Reservoir { free: 100.0, bound: 0.0 };
    let crystal = Crystal { capacity: 80.0, binding_rate: 0.5, release_rate: 0.03 };
    let initial = state.total();
    for _ in 0..1000 {
        step_crystal(&mut state, crystal, 0.02).expect("valid crystal model");
    }
    println!(
        "free={:.6}, bound={:.6}, conservation_error={:.12}",
        state.free, state.bound, (state.total() - initial).abs()
    );
    let mut field = vec![100.0, 0.0, 0.0, 0.0, 0.0];
    for _ in 0..100 {
        diffuse_1d(&mut field, 0.25, 1.0, 0.1).expect("stable diffusion step");
    }
    println!("mana field: {field:?}");
}

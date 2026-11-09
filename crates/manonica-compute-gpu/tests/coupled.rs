use manonica_compute_gpu::GpuCoupled;
use manonica_core::Quantity;
use manonica_materials::{ManaChannel, MaterialLattice, TrapMaterial};
use manonica_physics::{
    FreeManaField, GridShape, LocalBoundarySite, LocalPotentialBoundary, PotentialBoundary,
    PotentialField,
};
use manonica_simulation::{CoupledState, FieldControl, FieldLedger, MediumFlow, StepParameters};

fn scene(potential: bool) -> (CoupledState, StepParameters) {
    let shape = GridShape::new(4, 3, 2).unwrap();
    let mut field =
        FreeManaField::uniform(shape, 0.5, Quantity::nonnegative(4.0).unwrap()).unwrap();
    field
        .set(0, 0, 0, Quantity::nonnegative(20.0).unwrap())
        .unwrap();
    let mut materials = MaterialLattice::new(shape);
    for at in [[0, 0, 0], [1, 0, 0], [2, 0, 0]] {
        materials
            .install(
                at,
                TrapMaterial {
                    capacity: 0.7,
                    capture_rate: 1.0,
                    release_rate: 0.2,
                    binding_energy: 0.4,
                    heat_capacity: 2.0,
                    reference_temperature: 300.0,
                    activation_temperature: 100.0,
                    max_flux: 0.5,
                    conductivity: 1.0,
                },
            )
            .unwrap();
    }
    let mut storage = vec![[0.0; 2]; shape.len()];
    storage[0] = [0.2, 1.0];
    storage[1] = [0.1, 0.2];
    materials.try_replace_storage(&storage).unwrap();
    for (from, to) in [
        ([0, 0, 0], [3, 2, 1]),
        ([0, 0, 0], [1, 0, 0]),
        ([1, 0, 0], [2, 0, 0]),
    ] {
        materials
            .add_channel(ManaChannel {
                from,
                to,
                length: 1.2,
                area: 0.5,
                conductance: 0.3,
                max_flux: 1.0,
            })
            .unwrap();
    }
    let potential = potential.then(|| {
        let mut sources = vec![0.0; shape.len()];
        sources[0] = -1.0;
        sources[10] = 0.3;
        PotentialField::solve_with_local(
            shape,
            0.5,
            &sources,
            1.0,
            0.9,
            PotentialBoundary::Grounded,
            100,
            &[
                LocalBoundarySite {
                    index: 2,
                    condition: LocalPotentialBoundary::Dirichlet { value: -0.2 },
                },
                LocalBoundarySite {
                    index: 7,
                    condition: LocalPotentialBoundary::Neumann,
                },
                LocalBoundarySite {
                    index: 20,
                    condition: LocalPotentialBoundary::Robin {
                        reference: 0.1,
                        exchange: 0.3,
                    },
                },
            ],
        )
        .unwrap()
    });
    let field_ledger = potential.as_ref().map(|p| FieldLedger {
        field_energy_joules: p.field_energy_joules(),
        manon_potential_joules: p.manon_potential_energy(&field).unwrap()
            + materials.bound_potential_energy(p.values()).unwrap(),
        peak_potential: p.peak_absolute(),
        ..Default::default()
    });
    let parameters = StepParameters {
        diffusivity: 0.08,
        dt: 0.05,
        flow: MediumFlow::Vortex {
            origin: [0.0; 3],
            axis: [0.0, 1.0, 0.0],
            angular_speed: 0.1,
            axial_speed: 0.02,
        },
        thermal_conductance: 0.2,
        ambient_cooling_rate: 0.1,
        field: potential.as_ref().map(|_| FieldControl {
            mobility: 0.1,
            max_power_joules_per_second: 1000.0,
            shutdown_at_bound_fraction: 0.6,
            shutdown_decay_rate: 0.3,
        }),
    };
    (
        CoupledState {
            field,
            materials,
            potential,
            field_ledger,
            shutdown: false,
        },
        parameters,
    )
}
fn compare(cpu: &CoupledState, gpu: &CoupledState) {
    for (a, b) in cpu.field.densities().iter().zip(gpu.field.densities()) {
        assert!(
            (a.value() - b.value()).abs() < 1e-3,
            "free {} != {}",
            a.value(),
            b.value()
        );
    }
    for (a, b) in cpu.materials.cells().iter().zip(gpu.materials.cells()) {
        assert!(
            (a.bound - b.bound).abs() < 1e-3,
            "bound {} != {}",
            a.bound,
            b.bound
        );
        assert!(
            (a.heat - b.heat).abs() < 1e-3,
            "heat {} != {}",
            a.heat,
            b.heat
        );
    }
    let a = cpu.field_ledger.unwrap_or_default();
    let b = gpu.field_ledger.unwrap();
    for (name, x, y) in [
        ("field", a.field_energy_joules, b.field_energy_joules),
        ("manon", a.manon_potential_joules, b.manon_potential_joules),
        ("heat", a.dissipated_heat_joules, b.dissipated_heat_joules),
        ("work", a.operating_work_joules, b.operating_work_joules),
        ("exported", a.exported_heat_joules, b.exported_heat_joules),
    ] {
        assert!((x - y).abs() < 3e-3, "{name}: {x} != {y}");
    }
    assert_eq!(cpu.shutdown, gpu.shutdown);
}
#[test]
fn resident_coupled_operators_match_cpu_with_and_without_potential() {
    for potential in [false, true] {
        let (mut cpu, params) = scene(potential);
        let mut gpu = GpuCoupled::new(&cpu, &params).expect("coupled compute adapter required");
        for _ in 0..4 {
            for _ in 0..25 {
                cpu.step(&params).unwrap();
            }
            gpu.advance(25).unwrap();
            compare(&cpu, gpu.snapshot());
        }
        if potential {
            assert!(
                gpu.snapshot().shutdown,
                "test must exercise potential decay after saturation"
            );
        }
    }
}
#[test]
fn rejected_batch_preserves_snapshot_and_device_state() {
    let (initial, mut params) = scene(true);
    params.diffusivity = 10.0; // Deliberately unstable: device must reject it.
    let mut gpu = GpuCoupled::new(&initial, &params).unwrap();
    let before = gpu.snapshot().field.densities().to_vec();
    assert!(gpu.advance(3).is_err());
    assert_eq!(gpu.snapshot().field.densities(), before);
    assert!(gpu.advance(3).is_err()); // rollback includes device state and status
    assert_eq!(gpu.snapshot().field.densities(), before);
    assert!(gpu.advance(0).is_err());
}

#[test]
fn parallel_drift_subdivision_matches_reference() {
    let (mut cpu, mut params) = scene(true);
    params.field.as_mut().unwrap().mobility = 20.0;
    params.field.as_mut().unwrap().shutdown_at_bound_fraction = 0.0;
    let mut gpu = GpuCoupled::new(&cpu, &params).unwrap();
    for _ in 0..20 {
        cpu.step(&params).unwrap();
    }
    gpu.advance(20).unwrap();
    compare(&cpu, gpu.snapshot());
}

#[test]
fn hierarchical_reductions_cover_partial_workgroups() {
    let shape = GridShape::new(13, 7, 5).unwrap();
    let mut field =
        FreeManaField::uniform(shape, 0.5, Quantity::nonnegative(0.0).unwrap()).unwrap();
    for i in 0..shape.len() {
        field
            .set_flat(
                i,
                Quantity::nonnegative((i % 17 + 1) as f64 * 0.13).unwrap(),
            )
            .unwrap();
    }
    let (_, params) = scene(false);
    let mut cpu = CoupledState {
        field,
        materials: MaterialLattice::new(shape),
        potential: None,
        field_ledger: None,
        shutdown: false,
    };
    let mut gpu = GpuCoupled::new(&cpu, &params).unwrap();
    for _ in 0..20 {
        cpu.step(&params).unwrap();
    }
    gpu.advance(20).unwrap();
    compare(&cpu, gpu.snapshot());
}

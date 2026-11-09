use super::FormulaEditor;
use crate::experiment::ExperimentBuilder;
#[test]
fn idle_and_control_text_do_not_erase_a_selected_expression() {
    let mut editor = FormulaEditor::default();
    let original = editor.expressions[0].clone();
    editor.select_all();
    editor.input("");
    editor.input("\r\n");
    assert_eq!(editor.expressions[0], original);
    assert!(editor.selection_all);
    editor.input("3*cos(t)");
    assert_eq!(editor.expressions[0], "3*cos(t)");
    assert!(!editor.selection_all);
}
#[test]
fn cached_preflight_tracks_edits_and_reloaded_experiments() {
    let mut editor = FormulaEditor::default();
    let experiment = ExperimentBuilder::new("preview cache")
        .grid([12, 8, 12], 1.0)
        .formula(editor.formula())
        .build()
        .unwrap();
    editor.load(&experiment.formulas);
    editor.validate_physical_preview(&experiment);
    assert!(editor.preview_error.is_none());
    editor.validate_physical_preview(&experiment);
    assert!(editor.preview_error.is_none());
    editor.expressions[0] = "cos(".into();
    editor.validate_physical_preview(&experiment);
    assert!(editor.preview_error.is_some());
    editor.load(&experiment.formulas);
    editor.validate_physical_preview(&experiment);
    assert!(editor.preview_error.is_none());
}

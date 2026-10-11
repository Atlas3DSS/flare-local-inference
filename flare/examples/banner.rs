use std::{io::IsTerminal, thread, time::Duration};

use flare_helios::startup::StartupProgress;

fn main() {
    let mut progress = StartupProgress::stderr();
    if std::io::stderr().is_terminal() {
        thread::sleep(Duration::from_secs(3));
    }
    progress.finish_animation();
}

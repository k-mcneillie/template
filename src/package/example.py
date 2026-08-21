from __future__ import annotations

import torch
from sesh import Session

if __name__ == "__main__":
    print("=== Showcasing Complete Integrated Public Session API ===")

    # 1. Pipeline initialisation context block
    # This automatically provisions the directory, configures loggers, sets
    # deterministic seeds, and safely sets up isolated internal MLflow
    # tracks under the hood.
    target_hardware = "cuda" if torch.cuda.is_available() else "cpu"

    with Session(
        name="comprehensive_scientific_run", seed=8888, device=target_hardware
    ) as session:
        # 2. Plain-text information logging via the public wrapper shortcut
        session.info("Commencing execution showcase pipeline step 1: Setup validation.")

        # 3. Parameter logging
        # Synchronises configurations effortlessly across local run.log files and the
        # MLflow instance
        session.log_params(
            {
                "learning_rate_initial": 0.0005,
                "weight_decay_coefficient": 1e-4,
                "dataset_split_ratio": [0.8, 0.1, 0.1],
            }
        )

        # 4. Dataset Card generation
        # Generates structured documentation and registers parameters directly into
        # active session space
        session.info("Commencing execution showcase pipeline step 2: Data synthesis.")
        session.log_dataset_card(
            name="GaussianNoiseFieldSynthesiser",
            parameters={"spatial_resolution": 2048, "noise_floor_db": -60.0},
            description=[
                "High-resolution synthetic validation grid.",
                "Generated to evaluate convergence under extreme boundary conditions.",
            ],
        )

        # 5. Metric streaming loop
        # Sequentially prints to the terminal console, logs to file, and updates MLflow
        # real-time graphs
        session.info(
            "Commencing execution showcase pipeline step 3: Training emulation loops."
        )
        for epoch in range(1, 4):
            simulated_loss = 0.85 / (epoch**0.5)
            simulated_metric = 0.72 + (0.06 * epoch)

            session.log_metrics(
                metrics={
                    "objective_loss": simulated_loss,
                    "validation_accuracy": simulated_metric,
                },
                step=epoch,
            )

        # 6. Model Card generation
        # Finalises documentation, packages metadata, and links absolute environment
        # tracking states
        session.info(
            "Commencing execution pipeline step 4: Model preservation tracking."
        )
        session.log_model_card(
            name="AnomalyClassifierNet",
            architecture="ResNetBackbone_S3",
            parameters={"hidden_channels": 256, "dropout_probability": 0.3},
            intended_use=[
                "Predictive anomaly classification within non-uniform tensor grids."
            ],
            limitations=[
                "Degrades rapidly when processing input domains outside trained bounds."
            ],
            training_metadata={"epochs_executed": 3, "optimal_loss_attained": 0.490},
            description="Trained model optimised for structural matrix parsing.",
        )

    # Context manager automatically triggers clean resource cleanup and finalises
    # connections
    print(
        "\nExecution loop finalised successfully. "
        + f"Review all local run outputs at: {session.output_dir}"
    )
    print("=== Showcasing Completed ===")

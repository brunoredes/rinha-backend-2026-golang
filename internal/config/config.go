package config

import (
	"time"
)

type APIConfig struct {
	Addr string
	EngineAddr string
	EngineTimeout time.Duration
	Normalization string
	MCCRisk string
}

type EngineConfig struct {
	Addr string
	ArtifactsPath string
	NProbe int
}

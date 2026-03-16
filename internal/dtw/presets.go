package dtw

// Preset specifies attention heads for a model.
type Preset struct {
	Layer int   // encoder or decoder layer index
	Heads []int // head indices to average
}

// Presets maps model names to alignment heads.
var Presets = map[string]Preset{
	"tiny.en":       {Layer: 1, Heads: []int{0, 1}},
	"tiny":          {Layer: 1, Heads: []int{0, 1}},
	"base.en":       {Layer: 2, Heads: []int{0, 2}},
	"base":          {Layer: 2, Heads: []int{0, 2}},
	"small.en":      {Layer: 3, Heads: []int{0, 1, 4}},
	"small":         {Layer: 3, Heads: []int{0, 1, 4}},
	"medium.en":     {Layer: 4, Heads: []int{0, 1, 5, 6}},
	"medium":        {Layer: 4, Heads: []int{0, 1, 5, 6}},
	"large":         {Layer: 16, Heads: []int{0, 1, 8, 9, 16, 17}},
	"large-v1":      {Layer: 16, Heads: []int{0, 1, 8, 9, 16, 17}},
	"large-v2":      {Layer: 16, Heads: []int{0, 1, 8, 9, 16, 17}},
	"large-v3":      {Layer: 16, Heads: []int{0, 1, 8, 9, 16, 17}},
	"large-v3-turbo": {Layer: 8, Heads: []int{0, 2, 4, 6}},
}

// GetPreset returns the preset for the given model name.
func GetPreset(modelName string) (Preset, bool) {
	p, ok := Presets[modelName]
	return p, ok
}

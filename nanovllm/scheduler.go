package nanovllm

import "container/list"

// Scheduler manages sequence scheduling for prefill and decode phases
type Scheduler struct {
	maxNumSeqs          int
	maxNumBatchedTokens int
	eos                 int
	blockManager        *BlockManager
	waiting             *list.List
	running             *list.List
}

// NewScheduler creates a new scheduler
func NewScheduler(config *Config) *Scheduler {
	numBlocks := config.NumKVCacheBlocks
	if numBlocks == -1 {
		// Default: allocate reasonable number of blocks
		numBlocks = 1024
	}

	return &Scheduler{
		maxNumSeqs:          config.MaxNumSeqs,
		maxNumBatchedTokens: config.MaxNumBatchedTokens,
		eos:                 config.EOS,
		blockManager:        NewBlockManager(numBlocks, config.KVCacheBlockSize),
		waiting:             list.New(),
		running:             list.New(),
	}
}

// IsFinished returns true if there are no more sequences to process
func (s *Scheduler) IsFinished() bool {
	return s.waiting.Len() == 0 && s.running.Len() == 0
}

// Add adds a sequence to the waiting queue
func (s *Scheduler) Add(seq *Sequence) {
	s.waiting.PushBack(seq)
}

// Schedule schedules sequences for the next step
// Returns the scheduled sequences and whether this is a prefill step
func (s *Scheduler) Schedule() ([]*Sequence, bool) {
	// Try prefill first
	scheduledSeqs := make([]*Sequence, 0)
	numSeqs := 0
	numBatchedTokens := 0

	for s.waiting.Len() > 0 && numSeqs < s.maxNumSeqs {
		elem := s.waiting.Front()
		seq := elem.Value.(*Sequence)

		if numBatchedTokens+seq.Len() > s.maxNumBatchedTokens || !s.blockManager.CanAllocate(seq) {
			break
		}

		numSeqs++
		s.blockManager.Allocate(seq)
		numBatchedTokens += seq.Len() - seq.NumCachedTokens
		seq.Status = StatusRunning

		s.waiting.Remove(elem)
		s.running.PushBack(seq)
		scheduledSeqs = append(scheduledSeqs, seq)
	}

	if len(scheduledSeqs) > 0 {
		return scheduledSeqs, true
	}

	// Decode phase
	for s.running.Len() > 0 && numSeqs < s.maxNumSeqs {
		elem := s.running.Front()
		seq := elem.Value.(*Sequence)
		s.running.Remove(elem)

		// Check if we can append
		for !s.blockManager.CanAppend(seq) {
			if s.running.Len() > 0 {
				// Preempt from the back
				lastElem := s.running.Back()
				lastSeq := lastElem.Value.(*Sequence)
				s.running.Remove(lastElem)
				s.preempt(lastSeq)
			} else {
				// Preempt current sequence
				s.preempt(seq)
				break
			}
		}

		// If not preempted, schedule it
		if seq.Status == StatusRunning {
			numSeqs++
			s.blockManager.MayAppend(seq)
			scheduledSeqs = append(scheduledSeqs, seq)
		}
	}

	if len(scheduledSeqs) == 0 {
		panic("no sequences scheduled")
	}

	// Put scheduled sequences back at the front of running queue
	for i := len(scheduledSeqs) - 1; i >= 0; i-- {
		s.running.PushFront(scheduledSeqs[i])
	}

	return scheduledSeqs, false
}

// preempt preempts a sequence
func (s *Scheduler) preempt(seq *Sequence) {
	seq.Status = StatusWaiting
	s.blockManager.Deallocate(seq)
	s.waiting.PushFront(seq)
}

// Postprocess processes the output tokens from model execution
func (s *Scheduler) Postprocess(seqs []*Sequence, tokenIDs []int) {
	for i, seq := range seqs {
		tokenID := tokenIDs[i]
		seq.AppendToken(tokenID)

		// Check if sequence is finished
		if (!seq.IgnoreEOS && tokenID == s.eos) || seq.NumCompletionTokens() == seq.MaxTokens {
			seq.Status = StatusFinished
			s.blockManager.Deallocate(seq)
			// Remove from running list
			for elem := s.running.Front(); elem != nil; elem = elem.Next() {
				if elem.Value.(*Sequence).SeqID == seq.SeqID {
					s.running.Remove(elem)
					break
				}
			}
		}
	}
}

package main

import (
	"context"
	"flag"
	"fmt"
	"log"
	"net"
	"sync"
	"time"

	"google.golang.org/grpc"
	"google.golang.org/grpc/credentials"
	"google.golang.org/grpc/credentials/insecure"
	"google.golang.org/grpc/metadata"
	"google.golang.org/protobuf/types/known/emptypb"

	pb "ML Recommendation Engine/protos/recommendation/v1"
)

var (
	port           = flag.Int("port", 50051, "gRPC server port")
	backendAddrs   = flag.String("backends", "localhost:50052,localhost:50053,localhost:50054", "Comma-separated backend addresses")
	maxConns       = flag.Int("max-connections", 100, "Maximum connections per backend")
	requestTimeout = flag.Int("timeout-ms", 50, "Request timeout in milliseconds")
	enableTLS      = flag.Bool("tls", false, "Enable TLS")
	certFile       = flag.String("cert", "cert.pem", "TLS certificate file")
	keyFile       = flag.String("key", "key.pem", "TLS key file")
)

// SidecarService implements the gRPC forwarding service
type SidecarService struct {
	pb.UnimplementedRecommendationServiceServer
	backends      []string
	pool         *ConnectionPool
	mu           sync.RWMutex
	stats        Stats
}

// Stats tracks sidecar metrics
type Stats struct {
	RequestsReceived   int64
	RequestsFailed     int64
	RequestsSucceeded int64
	LatencySum        int64
	ActiveRequests    int64
}

func NewSidecarService(backends []string, maxConns int) *SidecarService {
	return &SidecarService{
		backends: backends,
		pool:     NewConnectionPool(backends, maxConns),
	}
}

func main() {
	flag.Parse()

	log.Printf("Starting gRPC sidecar on port %d", *port)
	log.Printf("Backend addresses: %s", *backendAddrs)

	backends := parseBackends(*backendAddrs)
	service := NewSidecarService(backends, *maxConns)

	lis, err := net.Listen("tcp", fmt.Sprintf(":%d", *port))
	if err != nil {
		log.Fatalf("Failed to listen: %v", err)
	}

	var opts []grpc.ServerOption
	if *enableTLS {
		creds, err := credentials.NewServerTLSFromFile(*certFile, *keyFile)
		if err != nil {
			log.Fatalf("Failed to load TLS credentials: %v", err)
		}
		opts = append(opts, grpc.Creds(creds))
	}

	grpcServer := grpc.NewServer(opts...)
	pb.RegisterRecommendationServiceServer(grpcServer, service)

	log.Println("gRPC sidecar server started")
	if err := grpcServer.Serve(lis); err != nil {
		log.Fatalf("Failed to serve: %v", err)
	}
}

func parseBackends(addrs string) []string {
	var backends []string
	for _, addr := range splitAndTrim(addrs, ",") {
		if addr != "" {
			backends = append(backends, addr)
		}
	}
	return backends
}

func splitAndTrim(s, sep string) []string {
	var result []string
	start := 0
	for i := 0; i < len(s); i++ {
		if s[i:i+len(sep)] == sep {
			if start < i {
				result = append(result, s[start:i])
			}
			start = i + len(sep)
		}
	}
	if start < len(s) {
		result = append(result, s[start:])
	}
	return result
}

// GetRecommendations forwards request to backends with fan-out
func (s *SidecarService) GetRecommendations(ctx context.Context, req *pb.RecommendationRequest) (*pb.RecommendationResponse, error) {
	start := time.Now()
	s.mu.Lock()
	s.stats.RequestsReceived++
	s.stats.ActiveRequests++
	s.mu.Unlock()

	defer func() {
		s.mu.Lock()
		s.stats.ActiveRequests--
		s.mu.Unlock()
	}()

	// Fan-out to all backends
	respCh := make(chan *pb.RecommendationResponse, len(s.backends))
	errCh := make(chan error, len(s.backends))

	var wg sync.WaitGroup
	for _, backend := range s.backends {
		wg.AddChildProcess(func() {
			resp, err := s.forwardRequest(ctx, backend, req)
			if err != nil {
				errCh <- err
				return
			}
			respCh <- resp
		})
	}

	// Wait for first successful response or all errors
	doneCh := make(chan struct{})
	go func() {
		wg.Wait()
		close(doneCh)
	}()

	select {
	case resp := <-respCh:
		elapsed := time.Since(start)
		s.mu.Lock()
		s.stats.RequestsSucceeded++
		s.stats.LatencySum += elapsed.Milliseconds()
		s.mu.Unlock()
		return resp, nil

	case err := <-errCh:
		s.mu.Lock()
		s.stats.RequestsFailed++
		s.mu.Unlock()
		return nil, err

	case <-doneCh:
		return nil, fmt.Errorf("all backends failed")
	}
}

func (s *SidecarService) forwardRequest(ctx context.Context, backend string, req *pb.RecommendationRequest) (*pb.RecommendationResponse, error) {
	conn, err := s.pool.GetConnection(ctx, backend)
	if err != nil {
		return nil, fmt.Errorf("failed to get connection: %w", err)
	}
	defer s.pool.ReleaseConnection(backend, conn)

	client := pb.NewRecommendationServiceClient(conn)
	resp, err := client.GetRecommendations(ctx, req)
	if err != nil {
		return nil, fmt.Errorf("backend request failed: %w", err)
	}

	return resp, nil
}

// Health check
func (s *SidecarService) Health(ctx context.Context, req *emptypb.Empty) (*pb.HealthResponse, error) {
	return &pb.HealthResponse{
		Status: pb.HealthResponse_SERVING,
	}, nil
}

// Stats endpoint
func (s *SidecarService) Stats(ctx context.Context, req *emptypb.Empty) (*pb.StatsResponse, error) {
	s.mu.RLock()
	defer s.mu.RUnlock()

	avgLatency := int64(0)
	if s.stats.RequestsSucceeded > 0 {
		avgLatency = s.stats.LatencySum / s.stats.RequestsSucceeded
	}

	return &pb.StatsResponse{
		RequestsReceived:  s.stats.RequestsReceived,
		RequestsFailed:  s.stats.RequestsFailed,
		ActiveRequests: s.stats.ActiveRequests,
		AvgLatencyMs:    avgLatency,
	}, nil
}

// ConnectionPool manages a pool of connections to each backend
type ConnectionPool struct {
	backends  []string
	maxConns  int
	mu       sync.RWMutex
	conns    map[string][]*grpc.ClientConn
	overflow map[string]chan *grpc.ClientConn
}

func NewConnectionPool(backends []string, maxConns int) *ConnectionPool {
	return &ConnectionPool{
		backends:  backends,
		maxConns: maxConns,
		conns:    make(map[string][]*grpc.ClientConn),
		overflow: make(map[string]chan *grpc.ClientConn),
	}
}

func (p *ConnectionPool) GetConnection(ctx context.Context, backend string) (*grpc.ClientConn, error) {
	p.mu.Lock()
	defer p.mu.Unlock()

	// Check existing connections
	if conns, ok := p.conns[backend]; ok && len(conns) > 0 {
		conn := conns[len(conns)-1]
		p.conns[backend] = conns[:len(conns)-1]
		return conn, nil
	}

	// Create new connection
	var opts []grpc.DialOption
	opts = append(opts, grpc.WithTransportCredentials(insecure.NewCredentials()))
	opts = append(opts, grpc.WithKeepaliveParams(grpc KeepaliveServerParameters{
		Time:    20 * time.Second,
		Timeout: 10 * time.Second,
	}))

	conn, err := grpc.DialContext(ctx, backend, opts...)
	if err != nil {
		return nil, err
	}

	return conn, nil
}

func (p *ConnectionPool) ReleaseConnection(backend string, conn *grpc.ClientConn) {
	p.mu.Lock()
	defer p.mu.Unlock()

	if len(p.conns[backend]) < p.maxConns {
		p.conns[backend] = append(p.conns[backend], conn)
	} else {
		conn.Close()
	}
}